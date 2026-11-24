from langchain_chroma import Chroma
from gemini_embeddings import GeminiEmbeddings, client, LLM_MODEL

DB_DIR = "./chroma_gemini_db"
COLLECTION = "sichuan_routes_gemini"

db = Chroma(
    collection_name=COLLECTION,
    persist_directory=DB_DIR,
    embedding_function=GeminiEmbeddings(),
)
retriever = db.as_retriever(search_kwargs={"k": 4})

PROMPT = """You are a customer support agent for mountain guiding routes in western Sichuan, China.
Answer the customer's question using ONLY the information below.
If the answer is not in the information, say you don't have that detail.
Name the route you are talking about.

Information:
{information}

Customer question: {question}
"""


def answer(question: str) -> str:
    docs = retriever.invoke(question)
    information = "\n\n---\n\n".join(
        f"[Route: {d.metadata['route']}]\n{d.page_content}" for d in docs
    )
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=PROMPT.format(information=information, question=question),
    )
    return response.text


if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (q to quit): ")
        if question == "q":
            break
        print(answer(question))