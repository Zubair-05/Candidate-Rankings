from fastapi import FastAPI

from app.routers import health

app = FastAPI(
    title="Candidate Ranking API",
    description="Multi-signal AI candidate ranking pipeline",
    version="0.1.0",
)

app.include_router(health.router)
