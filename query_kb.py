from qdrant_client import QdrantClient
from openai import OpenAI
import tiktoken

# CONFIG
OPENAI_API_KEY = "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"
QDRANT_URL = "https://97d1f6e9-a2c9-4396-a58c-5807c7cc0a52.us-east4-0.gcp.cloud.qdrant.io/"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Wlzi6SDqW_yeNDWS9YLsxMjJU30t7PQSA9N8wyxmrUQ"
COLLECTION_NAME = "zendesk_kb"

openai = OpenAI(api_key=OPENAI_API_KEY)
tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False
)

def embed_query(query):
    res = openai.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    )
    return res.data[0].embedding

def search_kb(query, top_k=3):
    query_vector = embed_query(query)
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    return hits

def summarize_answer(query, context_chunks):
    context_text = "\n\n".join(chunk.payload["text"] for chunk in context_chunks)
    prompt = (
        f"You are a helpful AI assistant trained on customer support articles.\n\n"
        f"User question: {query}\n\n"
        f"Here are some relevant article excerpts:\n\n{context_text}\n\n"
        f"Based on this, provide a helpful and concise answer:"
    )
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return completion.choices[0].message.content.strip()

def continue_conversation(history, query, context_chunks):
    context_text = "\n\n".join(chunk.payload["text"] for chunk in context_chunks)
    messages = history + [
        {"role": "user", "content": f"{query}\n\nRelevant excerpts:\n{context_text}"}
    ]

    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.3
    )
    reply = completion.choices[0].message.content.strip()
    messages.append({"role": "assistant", "content": reply})
    return reply, messages
    


if __name__ == "__main__":
    chat_history = []
    query = input("🔍 Ask your question: ").strip()
    if not query:
        print("❌ You need to enter a question.")
        exit()

    results = search_kb(query)

    print("\n🧠 Top Matches:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.payload['title']} — {r.payload['url']}")
        print(r.payload["text"])
        print("-" * 80)

    answer = summarize_answer(query, results)
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

    print("\n🤖 GPT Summary:\n")
    print(answer)

    while True:
        follow_up = input("\n💬 Follow-up (or press Enter to exit): ").strip()
        if not follow_up:
            print("👋 Done. Goodbye!")
            break
        response, chat_history = continue_conversation(chat_history, follow_up, results)
        print("\n🤖", response)