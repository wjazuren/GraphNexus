from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from knowledge_agent import build_knowledge_agent
from knowledge_agent.factory import build_llm_from_env


app = FastAPI(title="OneKE Knowledge Agent API", version="1.0.0")
_agent = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    allow_web: bool = True


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_knowledge_agent(build_llm_from_env())
    return _agent


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/query")
async def query(request: QueryRequest):
    try:
        return await get_agent().ainvoke(request.question, allow_web=request.allow_web)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"knowledge agent unavailable: {type(exc).__name__}") from exc

