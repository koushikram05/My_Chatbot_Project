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
            # ✅ Extracted normal text
            text += page_text + "\n"
        else:
            # ⚠️ If no text, fallback to OCR
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

# ---------------------------
# Step 4: Embeddings
# ---------------------------
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# ---------------------------
# Step 5: Store in Qdrant
# ---------------------------
def store_in_qdrant(chunks, collection_name):
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
        embedding = get_embedding(chunk)
        points.append(models.PointStruct(
            id=i,
            vector=embedding,
            payload={"text": chunk}
        ))

    qdrant.upsert(collection_name=collection_name, points=points)
    print(f"✅ Stored {len(points)} chunks in {collection_name}")

# ---------------------------
# Chatbot
# ---------------------------
def chatbot_loop(collection_name):
    print("\n🤖 Chatbot is ready!")
    print("How can I assist you today?\n")

    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye! 👋")
            break

        query_vector = get_embedding(user_query)
        results = qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=3
        )

        if not results or not results.points:
            print("Bot: Sorry, I can’t find an answer related to your PDF.")
            continue

        context = " ".join([r.payload['text'] for r in results.points])

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that only answers based on the provided PDF content."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_query}"}
            ]
        )

        bot_reply = response.choices[0].message.content.strip()
        print(f"Bot: {bot_reply}")

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    file_path = "/Users/koushikramalingam/Desktop/Gen AI/Veg_Biryani.pdf"

    # Force collection name for Veg Biryani
    collection_name = "pdf_chunks_veg_biryani"

    text = read_pdf(file_path)
    tokens, enc = tokenize_text(text)
    chunks = create_chunks(tokens, enc, chunk_size=300, overlap=50)

    print(f"✅ Created {len(chunks)} chunks")
    store_in_qdrant(chunks, collection_name)

    # Start chatbot
    chatbot_loop(collection_name)
