from typing import List, Optional
from pydantic import BaseModel, Field

class CitationMetadata(BaseModel):
    pages_referenced: List[int] = Field(default_factory=list, description="Page numbers referenced")
    source_papyrus: str = Field(default="The Egyptian Book of the Dead", description="Source document identifier")

class ArchaeologicalAnalysis(BaseModel):
    spell_or_chapter: str = Field(..., description="Title, plate number, or chapter identifier (e.g. Chapter 30B)")
    deities: List[str] = Field(default_factory=list, description="Egyptian deities mentioned in the text")
    ritual_objective: str = Field(..., description="Primary magical or protective purpose of the text")
    sacred_objects_amulets: List[str] = Field(default_factory=list, description="Sacred artifacts, amulets, or offerings described")
    mythological_summary: str = Field(..., description="Religious or mythological belief behind the ritual")
    transliteration_terms: List[str] = Field(default_factory=list, description="Key transliterated Egyptian terms (e.g., Ka, Ba, Ab, Ma'at)")
    citations: CitationMetadata = Field(default_factory=CitationMetadata, description="Referenced source and pages")
    faithfulness_score: Optional[float] = Field(default=None, description="Groundedness confidence score")
