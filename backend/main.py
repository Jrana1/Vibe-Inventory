from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.core.ai_config import settings
from backend.core.database import engine
from backend.routers import ai, dashboard, health, inventory
from backend.services import ai_comparison, ai_store, ai_tracing


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings().enabled:
        ai_tracing.initialize()
        ai_store.AgentRun.__table__.create(engine, checkfirst=True)
        ai_store.AgentProposal.__table__.create(engine, checkfirst=True)
        ai_comparison.RouterComparison.__table__.create(engine, checkfirst=True)
        with Session(engine) as database:
            ai_store.recover_interrupted_runs(database)
            ai_comparison.recover(database)
    try:
        yield
    finally:
        ai_tracing.shutdown()


app = FastAPI(title="Vibe Inventory API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(inventory.router)
app.include_router(dashboard.router)
app.include_router(ai.router)
