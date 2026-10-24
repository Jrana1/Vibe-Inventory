from typing import Annotated, cast

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.schemas.dashboard import DashboardResponse
from backend.services.dashboard_service import dashboard as get_dashboard

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])
Database = Annotated[Session, Depends(get_db)]


@router.get("", response_model=DashboardResponse)
def dashboard(database: Database) -> DashboardResponse:
    return cast(DashboardResponse, DashboardResponse.model_validate(get_dashboard(database)))
