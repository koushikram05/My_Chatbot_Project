import os
import json
import redis
from dotenv import load_dotenv

# ---------------------------
# Step 1: Load environment variables
# ---------------------------
load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_USER = os.getenv("REDIS_USER")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

print("Loaded from env:")
print("REDIS_HOST:", REDIS_HOST)
print("REDIS_PORT:", REDIS_PORT)
print("REDIS_USER:", REDIS_USER)
print("REDIS_PASSWORD:", "***" if REDIS_PASSWORD else "MISSING")

# ---------------------------
# Step 2: Connect to Redis
# ---------------------------
try:
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        username=REDIS_USER,
        password=REDIS_PASSWORD,
        decode_responses=True
    )

    # Quick ping test
    if r.ping():
        print("✅ Connected to Redis!")
except Exception as e:
    print("❌ Redis connection failed:", str(e))
    exit(1)

# ---------------------------
# Step 3: Store sample FAQ
# ---------------------------
faq_key = "faq:rome:best_time"
faq_data = {
    "question": "What is the best time to visit Rome?",
    "answer": "Spring (April–June) and Fall (September–October) are best.",
    "embedding": [0.01, -0.02, 0.03, 0.04]  # Dummy short vector
}

r.set(faq_key, json.dumps(faq_data))
print(f"✅ Stored FAQ in Redis under key: {faq_key}")

# ---------------------------
# Step 4: Retrieve FAQ
# ---------------------------
retrieved = r.get(faq_key)
if retrieved:
    faq_loaded = json.loads(retrieved)
    print("Retrieved from Redis:", faq_loaded)
else:
    print("❌ No data found for key:", faq_key)

# ---------------------------
# Step 5: Clean-up (optional)
# ---------------------------
# Uncomment if you want to remove after test
# r.delete(faq_key)
# print(f"🗑️ Deleted key {faq_key}")
