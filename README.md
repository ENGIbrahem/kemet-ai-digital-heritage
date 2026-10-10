# 🏺 Kemet-AI: Digital Heritage Intelligence Platform
> **An Archaeological Retrieval-Augmented Generation (RAG) System for Structuring, Verifying, and Archiving Ancient Egyptian Funerary Papyri.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-green.svg)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/VectorStore-FAISS-purple.svg)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Quantization](https://img.shields.io/badge/Quantization-4--bit%20NF4-yellow.svg)](https://huggingface.co/)

---

## 🏛️ Overview & Problem Statement
Deciphering and archiving ancient Egyptian funerary literature (such as the *Book of the Dead*) presents unique challenges for researchers, curators, and museum catalogers:
* **Unstructured & Dispersed Texts:** Rituals, divine spells, and mythological narratives are distributed across hundreds of papyrus chapters and plates.
* **Complex Multi-entity Nomenclature:** Identifying deities, amulets, spells, and transliterated linguistic markers requires deep domain expertise.
* **Generative Hallucinations:** Standard large language models often hallucinate deities and ritual functions when querying esoteric historical contexts.

**Kemet-AI** bridges this gap by marrying dense semantic retrieval (**FAISS**) with strict schema validation (**Pydantic**) and a dual inference engine (**Google Gemini Cloud** & **Local 4-bit Quantized Qwen2.5**). It converts complex funerary texts into structured, citation-backed digital museum records in real time.

---

## ⚡ Key Features

* **Dual Inference Pipeline (Cloud & Local):**
  * **Google Gemini Cloud (`gemini-3.8-flash`):** High-throughput contextual synthesis for complex multi-chapter inquiries.
  * **Edge Quantized Model (`Qwen2.5-1.5B-Instruct` via 4-bit NF4):** Low-latency, privacy-preserving execution on budget GPU hardware with BitsAndBytes.
* **Strict Pydantic Schema Parsing:** Eliminates generative fluff and forces the LLM to output consistent museum-grade JSON records.
* **Page-Level Grounded Citations:** Tracks original document metadata to provide exact page-level evidence for every extracted entity.
* **Faithfulness & Groundedness Metric:** An automated grounding verification heuristic that scores the factual presence of extracted entities against retrieved source chunks.
* **Hieroglyphic Transliteration Extraction:** Identifies core Ancient Egyptian spiritual concepts (*Ka*, *Ba*, *Ab*, *Ma'at*).
* **On-the-Fly Document Ingestion:** Built-in PDF uploader allowing users to vectorize and index custom papyri dynamically.
* **Multi-Format Curatorial Export:** Single-click export to **Digital Museum JSON** and academic **BibTeX** format.
* **Royal Egyptian Glassmorphism UI:** Streamlit interface customized with ancient Egyptian color schemes, dark mode gold accents, and entity tagging.

---

## 📐 System Architecture

```text
[ User Query / Custom PDF ]
             │
             ▼
[ Document Ingestion & Chunking ] (RecursiveCharacterTextSplitter: 700 chars, 120 overlap)
             │
             ▼
[ Dense Vector Embedding ] (sentence-transformers/all-MiniLM-L6-v2)
             │
             ▼
[ FAISS Vector Store ] ──► [ Top-K Context + Page Metadata ]
                                        │
                                        ▼
                 [ Dual-Engine Archaeological Pipeline ]
                 ├── Engine A: Google Gemini 3.8-Flash (Cloud)
                 └── Engine B: Qwen2.5-1.5B 4-bit NF4 (Local GPU)
                                        │
                                        ▼
                         [ Pydantic Output Parser ]
                                        │
                                        ▼
                  [ Grounding & Faithfulness Verification ]
                                        │
                                        ▼
           [ Streamlit Presentation & Multi-Format Archival Export ]
           ├── Interactive Royal Dashboard
           ├── Museum-ready JSON
           └── Academic BibTeX
