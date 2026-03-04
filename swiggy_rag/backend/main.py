import os
from dotenv import load_dotenv
load_dotenv()
from rag.pdf_to_text import pdf_to_text
from rag.chunking import chunk_text
from rag.embed_store import build_and_save_index
from rag.rag_answer import retrieve, generate_answer

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

PDF_PATH = os.path.join(DATA_DIR, "swiggy_report.pdf")


def ingest():
    """Process PDF and build ChromaDB index."""
    print("Reading PDF...")
    text = pdf_to_text(PDF_PATH)

    print("Chunking text...")
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")

    print("Creating embeddings & ChromaDB index...")
    build_and_save_index(chunks)

    print("Ingestion complete!")


def chat():
    """CLI chat loop."""
    print("\nSwiggy Annual Report QA Bot")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ")

        if query.lower() in ["exit", "quit"]:
            break

        retrieved = retrieve(query)
        answer = generate_answer(query, retrieved)

        print("\nAnswer:")
        print(answer)

        print("\nSupporting Context:")
        for i, chunk in enumerate(retrieved, 1):
            print(f"\n--- Context {i} ---\n{chunk[:500]}...")


if __name__ == "__main__":
    print("1 Ingest PDF")
    print("2 Chat with bot")

    choice = input("Choose option: ")

    if choice == "1":
        ingest()
    elif choice == "2":
        chat()
    else:
        print("Invalid choice")
