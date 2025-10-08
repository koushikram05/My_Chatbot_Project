import os
import PyPDF2
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# ---------------------------
# Step 1: Load environment variables
# ---------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not OPENAI_API_KEY or not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("❌ Missing API keys or Qdrant config in .env")

print("✅ Keys loaded successfully")

# Initialize OpenAI + Qdrant client
client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ---------------------------
# Step 2: PDF Reading (with OCR fallback)
# ---------------------------
def read_pdf(file_path):
    pdf_reader = PyPDF2.PdfReader(file_path)
    text = ""

    for i, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            text += page_text + "\n"
        else:
            print(f"🔎 Page {i+1}: No text found, using OCR...")
            images = convert_from_path(file_path, first_page=i+1, last_page=i+1)
            for img in images:
                ocr_text = pytesseract.image_to_string(img)
                text += ocr_text + "\n"

    return text

# ---------------------------
# Step 3: Tokenization + Chunking
# ---------------------------
def tokenize_text(text):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return tokens, enc

def create_chunks(tokens, enc, chunk_size=300, overlap=50):
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i+chunk_size]
        chunks.append(enc.decode(chunk))
    return chunks

def create_large_chunks(tokens, enc, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i+chunk_size]
        chunks.append(enc.decode(chunk))
    return chunks

# ---------------------------
# Step 4: Embeddings
# ---------------------------
def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

# ---------------------------
# Step 5: Store in Qdrant
# ---------------------------
def store_in_qdrant(chunks, collection_name, model="text-embedding-3-small"):
    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1536,
                distance=models.Distance.COSINE
            )
        )
        print(f"✅ Created new collection: {collection_name}")

    points = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk, model=model)
        points.append(models.PointStruct(
            id=i,
            vector=embedding,
            payload={"text": chunk}
        ))

    qdrant.upsert(collection_name=collection_name, points=points)
    print(f"✅ Stored {len(points)} chunks in {collection_name}")

# ---------------------------
# Step 6: Fusion Retrieval
# ---------------------------
def fusion_retrieve(query, collection_names, top_k=3):
    all_results = []

    query_vector = get_embedding(query)

    for name in collection_names:
        results = qdrant.query_points(
            collection_name=name,
            query=query_vector,
            limit=top_k
        )
        if results and results.points:
            for r in results.points:
                all_results.append(r.payload["text"])

    # Deduplicate + keep diverse passages
    unique_contexts = list(dict.fromkeys(all_results))
    fused_context = " ".join(unique_contexts)

    return fused_context

# ---------------------------
# Chatbot with Fusion RAG
# ---------------------------
def chatbot_loop_fusion(collection_names):
    print("\n🤖 Fusion RAG Chatbot is ready!")
    print("How can I assist you today?\n")

    while True:
        user_query = input("You: ").strip()

        if not user_query:
            continue

        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye! 👋")
            break

        # Step 1: Retrieve fused context
        context = fusion_retrieve(user_query, collection_names, top_k=3)

        if not context:
            print("Bot: Sorry, I couldn’t find anything relevant in the PDF.")
            continue

        # Step 2: Generate answer using fused context
        final_prompt = f"""
        Context from PDF (fusion of multiple retrievals):
        {context}

        User Question: {user_query}

        Provide a clear, accurate answer grounded ONLY in the context above.
        """
        final_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that must base answers only on the provided PDF context."},
                {"role": "user", "content": final_prompt}
            ],
            max_tokens=400
        )
        final_answer = final_response.choices[0].message.content.strip()

        print(f"\n✅ Fusion Bot: {final_answer}\n")

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    file_path = "/Users/koushikramalingam/Desktop/Gen AI/Veg_Biryani.pdf"

    # Create two different collections for diversity
    collection_small = "pdf_chunks_biryani_small"
    collection_large = "pdf_chunks_biryani_large"

    text = read_pdf(file_path)
    tokens, enc = tokenize_text(text)

    # Two types of chunking
    small_chunks = create_chunks(tokens, enc, chunk_size=300, overlap=50)
    large_chunks = create_large_chunks(tokens, enc, chunk_size=500, overlap=100)

    print(f"✅ Created {len(small_chunks)} small chunks and {len(large_chunks)} large chunks")

    # Store in two separate collections
    store_in_qdrant(small_chunks, collection_small, model="text-embedding-3-small")
    store_in_qdrant(large_chunks, collection_large, model="text-embedding-3-small")

    # Start chatbot with fusion
    chatbot_loop_fusion([collection_small, collection_large])
