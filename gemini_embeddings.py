import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-3.6-flash"


class GeminiEmbeddings(Embeddings):

    BATCH_SIZE = 10
    PAUSE_SECONDS = 15

    def _embed(self, texts, task_type):
        vectors = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i:i + self.BATCH_SIZE]
            for attempt in range(6):
                try:
                    response = client.models.embed_content(
                        model=MODEL,
                        contents=batch,
                        config=types.EmbedContentConfig(task_type=task_type),
                    )
                    break
                except Exception as e:
                    if "429" not in str(e) or attempt == 5:
                        raise
                    print(f"Rate limited, waiting 60 seconds (attempt {attempt + 1})")
                    time.sleep(60)
            vectors.extend(item.values for item in response.embeddings)
            if len(texts) > self.BATCH_SIZE:
                time.sleep(self.PAUSE_SECONDS)
        return vectors

    def embed_documents(self, texts):
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text):
        return self._embed([text], "RETRIEVAL_QUERY")[0]