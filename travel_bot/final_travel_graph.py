# travel_graph.py
# Single-file travel bot: ingestion -> Qdrant -> LangGraph supervisor -> agents (RAG travel_agent + mock others)
import os
import sys
import time
import uuid
import json
import io
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# OpenAI official client
from openai import OpenAI

# Qdrant client
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# LangGraph
from langgraph.graph import StateGraph, END

# PDF/text reading
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# Google Drive (service account)
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2 import service_account
    GDRIVE_AVAILABLE = True
except Exception:
    GDRIVE_AVAILABLE = False

# Redis
try:
    import redis
except Exception:
    redis = None

# ---------------------------
# Config & logging
# ---------------------------
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("travel_graph")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("❌ ERROR: OPENAI_API_KEY missing in .env")
    sys.exit(1)

# Qdrant config
QDRANT_URL = os.getenv("QDRANT_URL")  # optional remote qdrant url
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # optional
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "travel_docs")

# Redis config
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_USER = os.getenv("REDIS_USER")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Google Drive config
GDRIVE_SA_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")  # path to service account JSON
GDRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")  # folder id to ingest

# ---------------------------
# Clients init
# ---------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# Qdrant init (local fallback)
QDRANT_DIR = PROJECT_DIR / "qdrant_local"
qdrant = None
try:
    if QDRANT_URL:
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        logger.info("Using remote Qdrant at %s", QDRANT_URL)
    else:
        qdrant = QdrantClient(path=str(QDRANT_DIR))
        logger.info("Using local Qdrant at %s", QDRANT_DIR)
except Exception as e:
    logger.error("Failed to initialize Qdrant client: %s", e)
    qdrant = None

# Redis init (optional)
redis_client = None
if redis and REDIS_HOST and REDIS_PORT:
    try:
        rp = int(REDIS_PORT)
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=rp,
            username=REDIS_USER,
            password=REDIS_PASSWORD,
            decode_responses=True
        )
        redis_client.ping()
        logger.info("Connected to Redis at %s:%s", REDIS_HOST, REDIS_PORT)
    except Exception as e:
        logger.warning("Redis connection failed: %s", e)
        redis_client = None
else:
    logger.info("Redis not configured or redis library missing; FAQ disabled.")

# ---------------------------
# Ensure Qdrant collection exists
# ---------------------------
def ensure_collection(name: str, dim: int = 1536):
    if qdrant is None:
        return
    try:
        if hasattr(qdrant, "collection_exists"):
            if not qdrant.collection_exists(name):
                qdrant.create_collection(
                    collection_name=name,
                    vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
                )
        else:
            try:
                qdrant.get_collection(name)
            except Exception:
                qdrant.recreate_collection(
                    collection_name=name,
                    vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
                )
        logger.info("Qdrant collection ready: %s", name)
    except Exception as e:
        logger.warning("Could not ensure Qdrant collection: %s", e)

ensure_collection(QDRANT_COLLECTION)

# ---------------------------
# Text chunking + file reading
# ---------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    text = text.replace("\r", " ").replace("\n", " ").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        end = min(i + chunk_size, L)
        chunks.append(text[i:end].strip())
        i = end - overlap
        if i <= 0:
            i = end
    return chunks

def read_pdf_text(path: Path) -> str:
    if PyPDF2 is None:
        logger.warning("PyPDF2 not installed; skipping PDF: %s", path)
        return ""
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages = []
            for p in reader.pages:
                try:
                    pages.append(p.extract_text() or "")
                except Exception:
                    pages.append("")
            return "\n".join(pages)
    except Exception as e:
        logger.warning("Failed to read PDF %s: %s", path, e)
        return ""

def read_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return path.read_text(encoding="latin-1", errors="ignore")
        except Exception as e:
            logger.warning("Failed to read txt %s: %s", path, e)
            return ""

# ---------------------------
# Ingest local docs -> Qdrant
# ---------------------------
def ingest_docs_to_qdrant(docs_folder: Path):
    if qdrant is None:
        logger.warning("Qdrant not available — skipping ingestion.")
        return
    if not docs_folder.exists() or not docs_folder.is_dir():
        logger.info("No docs folder found at %s — skipping ingestion.", docs_folder)
        return

    items = []
    for file in sorted(docs_folder.iterdir()):
        if not file.is_file():
            continue
        if file.suffix.lower() == ".pdf":
            text = read_pdf_text(file)
        elif file.suffix.lower() == ".txt":
            text = read_txt(file)
        else:
            continue
        if not text.strip():
            continue
        chunks = chunk_text(text, chunk_size=800, overlap=120)
        for i, c in enumerate(chunks):
            items.append({
                "id": f"{file.stem}_{i}_{uuid.uuid4().hex[:8]}",
                "text": c,
                "source": file.name,
                "page": i + 1
            })
    if not items:
        logger.info("No document chunks found to ingest.")
        return

    logger.info("Embedding %d chunks with OpenAI...", len(items))
    batch_size = 64
    points = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        texts = [it["text"] for it in batch]
        try:
            emb_resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
            vectors = [d.embedding for d in emb_resp.data]
        except Exception as e:
            logger.error("Embedding API failed: %s", e)
            return
        for it, vec in zip(batch, vectors):
            payload = {"text": it["text"], "source": it["source"], "page": it["page"]}
            try:
                pt = qmodels.PointStruct(id=it["id"], vector=vec, payload=payload)
                points.append(pt)
            except Exception:
                points.append({"id": it["id"], "vector": vec, "payload": payload})

    logger.info("Upserting %d points into Qdrant...", len(points))
    try:
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
        logger.info("Ingested %d chunks to Qdrant.", len(points))
    except Exception as e:
        logger.error("Failed to upsert to Qdrant: %s", e)

# ---------------------------
# Google Drive ingestion -> Qdrant (service account)
# ---------------------------
def ingest_docs_from_gdrive(sa_json_path: Optional[str], folder_id: Optional[str]) -> List[Dict[str, str]]:
    docs = []
    if not sa_json_path or not folder_id:
        logger.info("GDrive not configured; skipping Drive ingestion.")
        return docs
    if not GDRIVE_AVAILABLE:
        logger.warning("googleapiclient not installed; GDrive ingestion skipped.")
        return docs
    if not Path(sa_json_path).exists():
        logger.warning("Service account JSON not found at %s; skipping Drive ingestion.", sa_json_path)
        return docs
    try:
        creds = service_account.Credentials.from_service_account_file(sa_json_path, scopes=["https://www.googleapis.com/auth/drive"])
        service = build("drive", "v3", credentials=creds)

        # List files in folder (limit to text/pdf)
        query = f"'{folder_id}' in parents and (mimeType='application/pdf' or mimeType='text/plain')"
        resp = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
        files = resp.get("files", [])
        logger.info("Found %d files in GDrive folder.", len(files))
        for f in files:
            file_id = f["id"]
            mime = f.get("mimeType", "")
            # download bytes
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            fh.seek(0)
            data = fh.read()
            text = ""
            if "pdf" in mime:
                if PyPDF2:
                    try:
                        reader = PyPDF2.PdfReader(io.BytesIO(data))
                        pages = []
                        for p in reader.pages:
                            pages.append(p.extract_text() or "")
                        text = "\n".join(pages)
                    except Exception:
                        text = data.decode("utf-8", errors="ignore")
                else:
                    text = data.decode("utf-8", errors="ignore")
            else:
                text = data.decode("utf-8", errors="ignore")
            docs.append({"id": file_id, "name": f["name"], "text": text})
    except Exception as e:
        logger.error("GDrive ingestion failed: %s", e)
    return docs

def index_drive_docs_to_qdrant(docs: List[Dict[str, str]]):
    if not docs or qdrant is None:
        return
    points = []
    logger.info("Embedding and upserting %d Drive docs...", len(docs))
    for d in docs:
        try:
            emb = client.embeddings.create(model="text-embedding-3-small", input=d["text"]).data[0].embedding
            pid = f"drive_{d['id']}_{uuid.uuid4().hex[:6]}"
            try:
                pt = qmodels.PointStruct(id=pid, vector=emb, payload={"text": d["text"], "source": d["name"]})
                points.append(pt)
            except Exception:
                points.append({"id": pid, "vector": emb, "payload": {"text": d["text"], "source": d["name"]}})
        except Exception as e:
            logger.warning("Embedding drive doc failed: %s", e)
    if points:
        try:
            qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
            logger.info("Upserted Drive docs to Qdrant.")
        except Exception as e:
            logger.error("Failed to upsert drive docs: %s", e)

# ---------------------------
# Qdrant search helper (RAG)
# ---------------------------
def qdrant_search_topk(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if qdrant is None:
        return []
    try:
        emb_resp = client.embeddings.create(model="text-embedding-3-small", input=query)
        qv = emb_resp.data[0].embedding
    except Exception as e:
        logger.warning("Embedding failed for search: %s", e)
        return []
    try:
        resp = qdrant.search(collection_name=QDRANT_COLLECTION, query_vector=qv, limit=top_k)
        results = []
        for item in resp:
            payload = getattr(item, "payload", None) or item.get("payload", {})
            results.append(payload)
        return results
    except Exception as e:
        logger.warning("Qdrant search error: %s", e)
        return []

# ---------------------------
# Redis FAQ helpers
# ---------------------------
import numpy as np

def _cosine(a: List[float], b: List[float]) -> float:
    a_np = np.array(a, dtype=float)
    b_np = np.array(b, dtype=float)
    denom = (np.linalg.norm(a_np) * np.linalg.norm(b_np))
    if denom == 0:
        return -1.0
    return float(np.dot(a_np, b_np) / denom)

FAQ_THRESHOLD = float(os.getenv("FAQ_SIM_THRESHOLD", 0.78))

def search_faq_in_redis(question: str) -> Optional[Dict[str, Any]]:
    if redis_client is None:
        return None
    try:
        emb = client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding
    except Exception as e:
        logger.warning("Embeddings failed for FAQ search: %s", e)
        return None
    try:
        best_score = -2.0
        best_entry = None
        for k in redis_client.scan_iter("faq:*"):
            data = redis_client.hgetall(k)
            if not data or "embedding" not in data:
                continue
            try:
                stored = json.loads(data["embedding"])
            except Exception:
                try:
                    stored = json.loads(data["embedding"].replace("'", '"'))
                except Exception:
                    continue
            score = _cosine(emb, stored)
            if score > best_score:
                best_score = score
                best_entry = {"key": k, "question": data.get("question"), "answer": data.get("answer"), "score": score}
        if best_entry and best_entry["score"] >= FAQ_THRESHOLD:
            return {"answer": best_entry["answer"], "score": best_entry["score"], "key": best_entry["key"], "source": f"Redis FAQ ({best_entry['key']})", "agent": "FAQ Agent"}
    except Exception as e:
        logger.warning("Redis FAQ search failure: %s", e)
    return None

def store_faq_in_redis(question: str, answer: str):
    if redis_client is None:
        return
    try:
        emb = client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding
        key = f"faq:{uuid.uuid4().hex[:8]}"
        redis_client.hset(key, mapping={
            "question": question,
            "answer": answer,
            "embedding": json.dumps(emb)
        })
        logger.info("Stored FAQ in Redis: %s", key)
    except Exception as e:
        logger.error("Failed to store FAQ in Redis: %s", e)

# ---------------------------
# Sub-agents (all accept query: str and return dict)
# ---------------------------
def travel_agent(query: str) -> Dict[str, Any]:
    logger.info("[Travel Agent] query=%s", query)
    hits = qdrant_search_topk(query, top_k=4)
    if not hits:
        prompt = f"Create a concise travel itinerary or tips for: {query}"
        try:
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.7)
            ans = resp.choices[0].message.content
        except Exception as e:
            ans = f"[OpenAI error: {e}]"
        return {"answer": ans, "sources": ["OpenAI GPT"], "agent": "Travel Agent"}
    context_parts = []
    sources = []
    for h in hits:
        txt = h.get("text", "")
        src = h.get("source", "local")
        pg = h.get("page", "?")
        context_parts.append(txt)
        sources.append(f"{src} (page {pg})")
    context = "\n\n".join(context_parts[:6])
    prompt = f"Use the following documents to answer clearly and concisely.\n\nContext:\n{context}\n\nQuestion: {query}"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a travel assistant. Use the provided documents to answer precisely."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=600
        )
        ans = resp.choices[0].message.content
    except Exception as e:
        logger.error("OpenAI chat error: %s", e)
        ans = f"[OpenAI error: {e}]"
    return {"answer": ans, "sources": list(dict.fromkeys(sources)), "agent": "Travel Agent"}

def flight_agent(query: str) -> Dict[str, Any]:
    logger.info("[Flight Agent] (mock) %s", query)
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":f"Give 3 sample flight options for: {query}"}], temperature=0.6)
        ans = resp.choices[0].message.content
    except Exception:
        ans = "Mock flights: Airline A $500 | Airline B $480 | Airline C $520"
    return {"answer": ans, "sources": ["Mock Flight Data"], "agent": "Flight Agent"}

def hotel_agent(query: str) -> Dict[str, Any]:
    logger.info("[Hotel Agent] (mock) %s", query)
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":f"Suggest 3 hotels for: {query}"}], temperature=0.6)
        ans = resp.choices[0].message.content
    except Exception:
        ans = "Mock hotels: Budget $80 | Mid $150 | Luxury $320"
    return {"answer": ans, "sources": ["Mock Hotel Data"], "agent": "Hotel Agent"}

def currency_agent(query: str) -> Dict[str, Any]:
    logger.info("[Currency Agent] (mock) %s", query)
    import re
    m = re.search(r'([\d,.]+)\s*([A-Za-z]{3})\s*(to|in)\s*([A-Za-z]{3})', query)
    if m:
        amt = float(m.group(1).replace(",", ""))
        frm = m.group(2).upper()
        to = m.group(4).upper()
        rate = 0.92 if frm == "USD" and to == "EUR" else 1.0
        converted = round(amt * rate, 2)
        ans = f"{amt} {frm} ≈ {converted} {to} (mock rate {rate})"
    else:
        ans = "Mock currency info: 1 USD ≈ 0.92 EUR"
    return {"answer": ans, "sources": ["Mock Currency Data"], "agent": "Currency Agent"}

def general_agent(query: str) -> Dict[str, Any]:
    logger.info("[General Agent] calling OpenAI for: %s", query)
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":query}], temperature=0.7)
        ans = resp.choices[0].message.content
    except Exception as e:
        logger.error("OpenAI error: %s", e)
        ans = f"[OpenAI error: {e}]"
    return {"answer": ans, "sources": ["OpenAI GPT"], "agent": "General Agent"}

# ---------------------------
# Supervisor router (returns node name)
# ---------------------------
def supervisor(state: str) -> str:
    q = state.lower()
    try:
        faq = search_faq_in_redis(q)
        if faq:
            return "faq"
    except Exception:
        pass
    if "flight" in q or ("from" in q and "to" in q):
        return "flight"
    if "hotel" in q or "stay" in q or "room" in q:
        return "hotel"
    if any(w in q for w in ["convert", "currency", "usd", "eur", "exchange", "rate"]):
        return "currency"
    if any(w in q for w in ["itinerary", "plan", "days in", "what to do", "travel", "rome", "visit"]):
        return "travel"
    return "general"

# ---------------------------
# LangGraph wiring
# ---------------------------
graph = StateGraph(str)

graph.add_node("travel", travel_agent)
graph.add_node("flight", flight_agent)
graph.add_node("hotel", hotel_agent)
graph.add_node("currency", currency_agent)
graph.add_node("general", general_agent)

# replace lambda with proper function to avoid duplicate search calls
def faq_node(q: str) -> Dict[str, Any]:
    res = search_faq_in_redis(q)
    if res:
        return {"answer": res["answer"], "sources": [res["source"]], "agent": "FAQ Agent"}
    return {}

graph.add_node("faq", faq_node)
graph.add_node("supervisor", supervisor)

graph.set_entry_point("supervisor")

graph.add_conditional_edges(
    "supervisor",
    supervisor,
    {
        "travel": "travel",
        "flight": "flight",
        "hotel": "hotel",
        "currency": "currency",
        "general": "general",
        "faq": "faq"
    },
)

for node in ["travel", "flight", "hotel", "currency", "general", "faq"]:
    graph.add_edge(node, END)

app = graph.compile()

# ---------------------------
# Helper: determine if query is "storable" FAQ (not greeting)
# ---------------------------
QUESTION_WORDS = {"what", "when", "where", "how", "why", "who", "which", "best", "time", "recommend", "need"}
def looks_like_faq_to_store(q: str) -> bool:
    q = q.strip().lower()
    if not q:
        return False
    # not just short greetings
    if q in {"hi", "hello", "hey", "hey there", "hiya"}:
        return False
    # must be longer than 3 tokens or contain question mark or interrogative
    if "?" in q:
        return True
    tokens = q.split()
    if len(tokens) <= 3:
        return False
    if any(w in q for w in QUESTION_WORDS):
        return True
    return False

# ---------------------------
# Main chat loop
# ---------------------------
def main():
    # 1) ingest local docs to Qdrant if present
    ingest_docs_to_qdrant(PROJECT_DIR / "docs")

    # 2) ingest Google Drive docs if configured
    if GDRIVE_SA_JSON and GDRIVE_FOLDER_ID:
        drive_docs = ingest_docs_from_gdrive(GDRIVE_SA_JSON, GDRIVE_FOLDER_ID)
        if drive_docs:
            index_drive_docs_to_qdrant(drive_docs)

    print("\n✨ Travel Planner Bot ✨")
    print("Supervisor routes: TRAVEL / GENERAL / CURRENCY / FLIGHT / HOTEL / FAQ")
    print("Put PDF/TXT files in ./docs then restart to re-ingest.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("👋 Bye!")
            break

        # Choose node via supervisor (quick check)
        next_node = supervisor(user_input.lower())
        try:
            # Use invoke to call specific entry node
            result = app.invoke(user_input, config={"entry_point": next_node})
        except Exception as e:
            logger.error("Graph invoke failed: %s", e)
            # fallback to direct call
            if next_node == "travel":
                result = travel_agent(user_input)
            elif next_node == "flight":
                result = flight_agent(user_input)
            elif next_node == "hotel":
                result = hotel_agent(user_input)
            elif next_node == "currency":
                result = currency_agent(user_input)
            elif next_node == "faq":
                res = search_faq_in_redis(user_input)
                result = {"answer": res["answer"], "sources": [res["source"]], "agent": "FAQ Agent"} if res else {}
            else:
                result = general_agent(user_input)

        # unify result extraction
        answer = result.get("answer") or (result.get("messages", [{}])[-1].get("content") if isinstance(result.get("messages"), list) else None)
        sources = result.get("sources") or (result.get("source") and [result.get("source")]) or []
        agent_used = result.get("agent") or next_node

        if answer:
            # Echo the user's prompt inside the response block so it's always visible
            print("\n🤖 Travel Bot:\n")
            print(f"🗣 User asked: {user_input}\n")
            print(answer)
            print("\n📖 Sources:", ", ".join(sources) if sources else "None")
            print("🛠 Agent Used:", agent_used, "\n")

            # Auto-learn: store into Redis if not from Redis and looks like a real FAQ
            if redis_client and not any("Redis" in s or "faq" in s.lower() for s in sources):
                if looks_like_faq_to_store(user_input):
                    try:
                        store_faq_in_redis(user_input, answer)
                    except Exception as e:
                        logger.warning("Failed to auto-store FAQ: %s", e)
        else:
            print("\n⚠️ No answer produced. Try rephrasing.\n")

if __name__ == "__main__":
    main()
