import os
import io
import fitz  # PyMuPDF
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from openai import OpenAI
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv(dotenv_path="/Users/koushikramalingam/Desktop/Gen AI/.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

if not all([OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY, DRIVE_FOLDER_ID]):
    raise ValueError("❌ Missing environment variables in .env file")

print("✅ Keys loaded successfully")

# -------------------------------
# Initialize clients
# -------------------------------
client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# -------------------------------
# Helper: extract text chunks
# -------------------------------
def extract_text_chunks(pdf_path, chunk_size=500):
    chunks = []
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text("text")
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            if chunk.strip():
                chunks.append(chunk)
    return chunks

# -------------------------------
# Helper: get embeddings
# -------------------------------
def get_embeddings(chunks):
    if not chunks:
        return []
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )
    return [item.embedding for item in response.data]

# -------------------------------
# Helper: ensure collection
# -------------------------------
def ensure_collection(qdrant, collection_name, vector_size):
    if not qdrant.collection_exists(collection_name=collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"✅ Created new collection: {collection_name}")

# -------------------------------
# Helper: store chunks in Qdrant
# -------------------------------
def store_in_qdrant(chunks, collection_name, source, filename):
    if not chunks:
        print(f"⚠️ Skipping {filename} — no text extracted.")
        return

    embeddings = get_embeddings(chunks)
    if not embeddings:
        print(f"⚠️ Skipping {filename} — no embeddings generated.")
        return

    embedding_dim = len(embeddings[0])
    ensure_collection(qdrant, collection_name, embedding_dim)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload={"text": chunks[i], "source": source, "filename": filename}
        )
        for i in range(len(chunks))
    ]

    qdrant.upsert(collection_name=collection_name, points=points)
    print(f"✅ Inserted {len(chunks)} chunks from {filename} ({source})")

# -------------------------------
# Google Drive Authentication
# -------------------------------
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
creds = None
token_path = "token.pickle"

if os.path.exists(token_path):
    with open(token_path, "rb") as token:
        creds = pickle.load(token)

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json", SCOPES
        )
        creds = flow.run_local_server(port=0)
    with open(token_path, "wb") as token:
        pickle.dump(creds, token)

service = build("drive", "v3", credentials=creds)

# -------------------------------
# Main: ingest PDFs from Drive
# -------------------------------
def ingest_drive_pdfs():
    query = f"'{DRIVE_FOLDER_ID}' in parents and mimeType='application/pdf'"
    results = service.files().list(q=query).execute()
    items = results.get("files", [])

    if not items:
        print("⚠️ No PDFs found in Google Drive folder.")
        return

    print(f"📂 Found {len(items)} PDFs in Google Drive folder.")
    collection_name = "fusion_recipes_chunks"

    for file in items:
        file_id = file["id"]
        file_name = file["name"]

        print(f"⬇️ Downloading {file_name}...")
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()

        fh.seek(0)
        with open(file_name, "wb") as f:
            f.write(fh.read())

        chunks = extract_text_chunks(file_name)
        print(f"📖 Extracted {len(chunks)} chunks from {file_name}")

        if not chunks:
            print(f"⚠️ No text found in {file_name}, skipping ingestion.")
            continue

        store_in_qdrant(chunks, collection_name, source="drive", filename=file_name)

if __name__ == "__main__":
    ingest_drive_pdfs()
