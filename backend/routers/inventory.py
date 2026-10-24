from typing import Annotated, Literal, cast

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.schemas.inventory import (
    BulkPriceUpdate,
    BulkPriceUpdateResponse,
    InventoryItemResponse,
    InventoryListResponse,
    QuantityUpdate,
)
from backend.services import inventory_service

router = APIRouter(prefix="/api/inventory", tags=["inventory"])
Database = Annotated[Session, Depends(get_db)]


@router.get("", response_model=InventoryListResponse)
def list_inventory(
    database: Database,
    search: str = Query(default=""),
    category: str = Query(default=""),
    supplier: str = Query(default=""),
    stock_status: Literal["", "in-stock", "low-stock", "out-of-stock"] = "",
    sort: Literal["name", "price", "stock", "value"] = "name",
    order: Literal["asc", "desc"] = "asc",
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
) -> InventoryListResponse:
    result = inventory_service.list_inventory(
        database,
        search=search,
        category=category,
        supplier=supplier,
        stock_status=stock_status,
        sort=sort,
        order=order,
        page=page,
        page_size=page_size,
    )
    return cast(InventoryListResponse, InventoryListResponse.model_validate(result))


@router.patch(
    "/{item_id}/quantity",
    response_model=InventoryItemResponse,
    responses={404: {"description": "Inventory item not found"}},
)
def update_quantity(
    item_id: int, update: QuantityUpdate, database: Database
) -> InventoryItemResponse:
    item = inventory_service.update_quantity(database, item_id, update)
    return cast(InventoryItemResponse, InventoryItemResponse.model_validate(item))


@router.patch(
    "/bulk-price",
    response_model=BulkPriceUpdateResponse,
    status_code=status.HTTP_200_OK,
    responses={400: {"description": "Price adjustment is invalid"}},
)
def bulk_price_update(
    update: BulkPriceUpdate, database: Database
) -> BulkPriceUpdateResponse:
    result = inventory_service.bulk_price_update(database, update)
    return cast(BulkPriceUpdateResponse, BulkPriceUpdateResponse.model_validate(result))
