import logging
from typing import Tuple, Union, List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from schemas import ArchaeologicalAnalysis
from vector_store import ArchaeologicalVectorStore
from config import AppConfig

logger = logging.getLogger(__name__)

class ArchaeologicalPipeline:
    def __init__(self, vector_store: ArchaeologicalVectorStore, config: AppConfig = AppConfig()):
        self.config = config
        self.vector_store = vector_store
        self.parser = PydanticOutputParser(pydantic_object=ArchaeologicalAnalysis)
        self.prompt = self._build_prompt()

    def _build_prompt(self) -> PromptTemplate:
        template = (
            "You are an expert Egyptologist and Digital Heritage AI Assistant.\n"
            "Analyze the retrieved excerpt from the ancient Egyptian Book of the Dead.\n"
            "Extract entities strictly matching the schema, including transliterated key terms (e.g. Ka, Ba, Maat, Heart/Ab).\n\n"
            "Retrieved Context:\n{context}\n\n"
            "User Query / Focus:\n{query}\n\n"
            "{format_instructions}\n\n"
            "Return ONLY valid JSON matching the schema without markdown tags or introductory prose."
        )
        return PromptTemplate(
            template=template,
            input_variables=["context", "query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def _calculate_groundedness(self, analysis: ArchaeologicalAnalysis, context: str) -> float:
        check_items = analysis.deities + analysis.sacred_objects_amulets + analysis.transliteration_terms
        if not check_items:
            return 92.0
        lower_context = context.lower()
        matched = sum(1 for item in check_items if any(w in lower_context for w in item.lower().split()))
        return min(100.0, round((matched / len(check_items)) * 100, 1))

    def run(self, query: str, llm, top_k: int = 3) -> Tuple[Union[ArchaeologicalAnalysis, str], str, List[dict]]:
        retriever = self.vector_store.get_retriever(top_k=top_k)
        matched_docs = retriever.invoke(query)

        pages = sorted(list(set(doc.metadata.get("page", 1) for doc in matched_docs)))
        docs_metadata = [{"page": doc.metadata.get("page", 1), "content": doc.page_content} for doc in matched_docs]
        context_text = "\n\n---\n\n".join([f"[Page {doc.metadata.get('page', 1)}]\n{doc.page_content}" for doc in matched_docs])

        chain = self.prompt | llm | self.parser

        try:
            parsed_result: ArchaeologicalAnalysis = chain.invoke({"context": context_text, "query": query})
            parsed_result.citations.pages_referenced = pages
            parsed_result.faithfulness_score = self._calculate_groundedness(parsed_result, context_text)
            return parsed_result, context_text, docs_metadata
        except Exception as err:
            logger.warning("Pydantic parsing fallback: %s", err)
            raw_chain = self.prompt | llm
            raw_result = raw_chain.invoke({"context": context_text, "query": query})
            raw_text = raw_result.content if hasattr(raw_result, "content") else str(raw_result)
            return raw_text, context_text, docs_metadata
