import os
from pathlib import Path
from collections import defaultdict

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from gemini_embeddings import GeminiEmbeddings

DATA_DIR = "Data"
DB_DIR = "./chroma_gemini_db"
COLLECTION = "sichuan_routes_gemini"


def build():
    # 1. Load PDFs (one entry per page)
    pages = PyPDFDirectoryLoader(DATA_DIR).load()

    # 2. Group pages by PDF so each route is one document
    grouped = defaultdict(list)
    for page in pages:
        grouped[page.metadata["source"]].append(page.page_content)

    # 3. Split each route into chunks, and put the route name on EVERY chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    chunks = []
    for source, contents in grouped.items():
        route = Path(source).stem
        full_text = "\n".join(contents)
        for i, piece in enumerate(splitter.split_text(full_text)):
            chunks.append(Document(
                page_content=f"Route: {route}\n{piece}",
                metadata={"route": route, "source": Path(source).name, "chunk": i},
            ))

    print(f"Routes: {len(grouped)}")
    print(f"Chunks: {len(chunks)}")

    # 4. Store in Chroma with stable IDs, so rerunning doesn't duplicate chunks
    ids = [f"{c.metadata['route']}-{c.metadata['chunk']}" for c in chunks]

    db = Chroma(
        collection_name=COLLECTION,
        persist_directory=DB_DIR,
        embedding_function=GeminiEmbeddings(),
    )
    db.add_documents(chunks, ids=ids)

    print("Gemini embeddings successfully stored!")


if __name__ == "__main__":
    build()