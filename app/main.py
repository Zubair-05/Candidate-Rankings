import time
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Candidate Ranking API",
    description="Multi-signal AI candidate ranking pipeline",
    version="0.1.0",
)


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float


_start_time = time.time()


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version="0.1.0",
        uptime_seconds=round(time.time() - _start_time, 2),
    )
