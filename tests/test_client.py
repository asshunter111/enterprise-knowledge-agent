"""FastAPI TestClient test"""
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

# 1. Root
r = client.get("/")
data = r.json()
print(f"Root: {r.status_code} keys={list(data.keys())}")

# 2. Create session
r = client.post("/sessions", json={"title": "test"})
sid = r.json()["id"]
print(f"Session: {sid[:8]}...")

# 3. Upload doc
content = "CEO is Zhang, CTO is Li. Company has 500 employees. Revenue 2025: 200M."
r = client.post("/documents/upload", files={"file": ("test.txt", content.encode(), "text/plain")})
d = r.json()
print(f"Upload: {r.status_code} status={d['status']} chunks={d['chunk_count']}")
if d["status"] != "ready":
    print(f"  ERROR: {d.get('error_message')}")
    exit()

# 4. Chat
r = client.post("/chat", json={"session_id": sid, "query": "Who is the CEO"})
print(f"Chat: {r.status_code}")
if r.status_code == 200:
    a = r.json()
    print(f"  answer: {a['answer'][:200]}")
    print(f"  citations: {len(a['citations'])}")
else:
    print(f"  body: {r.text[:300]}")
    exit()

r = client.post("/chat", json={"session_id": sid, "query": "How many employees"})
print(f"Chat2: {r.status_code}")
if r.status_code == 200:
    print(f"  answer: {r.json()['answer'][:200]}")

print("\nAll tests passed!")
