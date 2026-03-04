from openai import OpenAI
from rag.embed_store import get_embedding, load_collection

CHAT_MODEL = "gpt-5.2"


def get_client():
    return OpenAI()


def retrieve(query, k=4):
    collection = load_collection()
    query_embedding = get_embedding([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results["documents"][0]


def generate_answer(question, retrieved_chunks):
    client = get_client()
    context = "\n\n".join(retrieved_chunks)

    response = client.responses.create(
        model=CHAT_MODEL,
        instructions=(
            "You are an AI assistant answering questions about the Swiggy Annual Report. "
            "Use ONLY the provided context. "
            "If the answer is not present, say: "
            "'I could not find this in the Swiggy Annual Report.'"
        ),
        input=f"Context:\n{context}\n\nQuestion:\n{question}",
    )

    return response.output_text
