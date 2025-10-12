# final_travel_graph_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from Final_travel_graph import app as langgraph_app, supervisor

app = FastAPI(
    title="Travel Planner Bot API",
    description="Travel chatbot API powered by LangGraph, OpenAI, and Amadeus sandbox.",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    agent: str
    sources: list

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    user_input = request.query.strip()
    # determine node
    node = supervisor(user_input)
    # invoke LangGraph
    try:
        result = langgraph_app.invoke(user_input, config={"entry_point": node})
        answer = result.get("answer") or ""
        agent = result.get("agent") or node
        sources = result.get("sources") or []
    except Exception as e:
        answer = f"[Error: {e}]"
        agent = node
        sources = []

    return ChatResponse(answer=answer, agent=agent, sources=sources)

@app.get("/health")
def health():
    return {"status": "ok"}
