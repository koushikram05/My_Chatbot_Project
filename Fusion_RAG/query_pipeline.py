import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI

# Load environment variables
load_dotenv(dotenv_path="/Users/koushikramalingam/Desktop/Gen AI/.env")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Qdrant + OpenAI clients
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

COLLECTION_NAME = "fusion_recipes_chunks"

# -------------------------------
# Embedding
# -------------------------------
def embed_text(text: str):
    """Create embedding for query/chunk using OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# -------------------------------
# Qdrant Search
# -------------------------------
def search_qdrant(query: str, top_k: int = 3):
    """Search Qdrant with query and return results w/ metadata."""
    query_vector = embed_text(query)
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    formatted = []
    for r in results:
        formatted.append({
            "text": r.payload.get("text", ""),
            "source": r.payload.get("source", "unknown"),
            "filename": r.payload.get("filename", "unknown"),
            "score": r.score
        })
    return formatted

# -------------------------------
# Intent Detection
# -------------------------------
def detect_intent(query: str):
    """
    Try LLM-based intent detection first.
    Fallback to keyword-based detection.
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Classify user query into: smalltalk, recipe, or other. Only output one label."},
                {"role": "user", "content": query}
            ],
            max_tokens=5,
        )
        intent = resp.choices[0].message.content.strip().lower()
        if intent in ["smalltalk", "recipe", "other"]:
            return intent
    except Exception:
        pass  # fallback if API fails

    # ---- Fallback: keyword-based ----
    smalltalk_keywords = ["hi", "hello", "hey", "how are you", "good morning"]
    recipe_keywords = ["cook", "recipe", "make", "ingredients", "prepare", "biryani", "raitha"]

    q_lower = query.lower()
    if any(k in q_lower for k in smalltalk_keywords):
        return "smalltalk"
    elif any(k in q_lower for k in recipe_keywords):
        return "recipe"
    else:
        return "other"

# -------------------------------
# Main Chat Loop
# -------------------------------
if __name__ == "__main__":
    print("🤖 Fusion RAG Chatbot Ready (with LLM intent detection)")

    while True:
        user_query = input("\nYou: ").strip()
        if user_query.lower() in ["quit", "exit", "bye"]:
            print("👋 Goodbye!")
            break

        intent = detect_intent(user_query)

        if intent == "smalltalk":
            print("\n💬 Smalltalk Response:")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a friendly assistant for smalltalk."},
                    {"role": "user", "content": user_query}
                ]
            )
            print(response.choices[0].message.content.strip())

        elif intent == "recipe":
            print("\n📌 Retrieved Results:\n")
            results = search_qdrant(user_query, top_k=3)
            for idx, res in enumerate(results, 1):
                print(f"{idx}. [{res['source']}] {res['filename']} (score={res['score']:.4f})")
                print(f"   {res['text']}\n")

        else:  # "other"
            print("\n🤔 General Response:")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer helpfully. If it's about recipes, say you can only answer with stored recipes."},
                    {"role": "user", "content": user_query}
                ]
            )
            print(response.choices[0].message.content.strip())
