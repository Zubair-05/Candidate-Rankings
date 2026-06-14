import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import health, ranking
from app.services import retrieval

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Building retrieval index…")
    retrieval.build_index()
    logger.info("Index ready — app is up")
    yield


app = FastAPI(
    title="Candidate Ranking API",
    description="Multi-signal AI candidate ranking pipeline",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(ranking.router)
