from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.logger import get_logger, log_latency
from app.routers import health, ranking
from app.services import retrieval

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with log_latency(logger, "startup_index_build"):
        retrieval.build_index()
    logger.info("Application ready — all systems up")
    yield


app = FastAPI(
    title="Candidate Ranking API",
    description="Multi-signal AI candidate ranking pipeline",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(ranking.router)

_static = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")
