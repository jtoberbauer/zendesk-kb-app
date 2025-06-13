import json
import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, CollectionStatus
from openai import OpenAI
import uuid

# CONFIG
OPENAI_API_KEY = "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"
QDRANT_URL = "https://97d1f6e9-a2c9-4396-a58c-5807c7cc0a52.us-east4-0.gcp.cloud.qdrant.io/"  # or your hosted Qdrant instance
COLLECTION_NAME = "zendesk_kb"
MAX_TOKENS = 800

client = QdrantClient(
    url=QDRANT_URL,
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Wlzi6SDqW_yeNDWS9YLsxMjJU30t7PQSA9N8wyxmrUQ",
    prefer_grpc=False
)
openai = OpenAI(api_key=OPENAI_API_KEY)
tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

def embed(text):
    res = openai.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return res.data[0].embedding

def chunk_text(text):
    words = text.split()
    chunks, chunk = [], []
    for word in words:
        chunk.append(word)
        if len(tokenizer.encode(" ".join(chunk))) > MAX_TOKENS:
            chunk.pop()
            chunks.append(" ".join(chunk))
            chunk = [word]
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

# Load the exported JSON
with open("zendesk_kb_articles.json") as f:
    articles = json.load(f)

# Create collection if needed
if not client.collection_exists(COLLECTION_NAME):
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

points = []
for article in articles:
    chunks = chunk_text(article["content"])
    for i, chunk in enumerate(chunks):
        vector = embed(chunk)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "title": article["title"],
                "url": article["url"],
                "chunk_index": i,
                "text": chunk
            }
        ))

# Upload to Qdrant
client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"✅ Uploaded {len(points)} chunks to Qdrant")
