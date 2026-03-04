# Swiggy_RAG_System
## RAG Based Question Answering System
A CLI-based Retrieval-Augmented Generation (RAG) system that enables semantic question answering over the Swiggy Annual Report. The pipeline performs PDF ingestion, text chunking, vector indexing, and context-aware response generation using Python, ChromaDB, and embedding-based retrieval, with optional LLM integration for answer synthesis.

## Data Source

- **Document:** Swiggy Annual Report (Latest Available)
- **Format:** PDF
- **Source:** Publicly available
- **Link:** https://www.swiggy.com/media/annual-report

The document is used exclusively for retrieval and answer generation.

## Project Objective
The objective of this project is to design and implement an AI system that:

- Answers user questions only from a given business document
- Prevents hallucination by grounding responses in retrieved context
- Demonstrates a real-world RAG architecture used in industry

##  Features

- PDF ingestion and text extraction
-  Intelligent token-based chunking
-  Semantic search using vector embeddings
-  ChromaDB vector store (Mac-friendly, no FAISS issues)
-  Context-aware answer generation
-  Offline embedding support (Sentence Transformers)
-  CLI chat interface
-  Hallucination-safe responses (answers only from report)

## Project Structure
swiggy-rag/
├── backend/
│   ├── main.py                 # CLI entry point
│   ├── rag/
│   │   ├── pdf_to_text.py      # PDF → text extraction
│   │   ├── chunking.py         # Text chunking logic
│   │   ├── embed_store.py      # Embeddings + ChromaDB
│   │   ├── rag_answer.py       # Retrieval + answer generation
│   └── data/
│       └── swiggy_report.pdf   # Source document
├── requirements.txt
└── README.md


## How to Run

```bash
1. clone the repo
git clone https://github.com/YOUR_USERNAME/swiggy-rag.git
cd swiggy-rag/backend

2. Create, activate virtual environment and install dependencies 
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
pip install -r ../requirements.txt

3. Set OpenAI API key
export OPENAI_API_KEY="your_api_key_here"

4. Run the app
python main.py

Choose an option
1 → Ingest PDF
2 → Chat with bot

```


# Note on OpenAI API Usage:
This project uses OpenAI embeddings and LLMs for document ingestion and question answering.
Running the ingestion step requires a valid OpenAI API key with active billing.
Due to API usage costs, the embedding step may not run locally without sufficient quota.
