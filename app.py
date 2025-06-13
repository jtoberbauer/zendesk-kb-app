import streamlit as st
from qdrant_client import QdrantClient
from openai import OpenAI
import tiktoken

# Config
OPENAI_API_KEY = "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"
QDRANT_URL = "https://97d1f6e9-a2c9-4396-a58c-5807c7cc0a52.us-east4-0.gcp.cloud.qdrant.io/"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Wlzi6SDqW_yeNDWS9YLsxMjJU30t7PQSA9N8wyxmrUQ"
COLLECTION_NAME = "zendesk_kb"

openai = OpenAI(api_key=OPENAI_API_KEY)
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)
tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

def embed_query(query):
    res = openai.embeddings.create(input=[query], model="text-embedding-3-small")
    return res.data[0].embedding

def search_kb(query, top_k=3):
    query_vector = embed_query(query)
    return client.search(collection_name=COLLECTION_NAME, query_vector=query_vector, limit=top_k)

def summarize_answer(query, chunks):
    context = "\n\n".join([c.payload["text"] for c in chunks])
    prompt = f"User question: {query}\n\nRelevant info:\n{context}\n\nAnswer clearly:"
    res = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return res.choices[0].message.content.strip()

# Streamlit UI
st.set_page_config(page_title="Zendesk AI Search", layout="wide")
st.title("🔎 Zendesk Knowledge Assistant")

# --- Question box ---
query = st.text_input("Ask a question about your support articles:")

if query:
    with st.spinner("Searching..."):
        results = search_kb(query)
        gpt_response = summarize_answer(query, results)

    # --- GPT Summary FIRST ---
    st.subheader("🤖 GPT Answer")
    st.markdown(f"<div style='background-color:#f0f2f6; padding:1em; border-radius:8px;'>{gpt_response}</div>", unsafe_allow_html=True)

    # --- Article Chunks ---
    st.subheader("🧠 Relevant KB Matches")
    for r in results:
        st.markdown(f"**{r.payload['title']}** — [View Original]({r.payload['url']})")
        st.markdown(f"<div style='background-color:#f9f9f9; padding:0.75em; border-radius:6px; word-wrap:break-word; overflow-wrap:break-word;'>{r.payload['text']}</div>", unsafe_allow_html=True)
        st.markdown("---")
