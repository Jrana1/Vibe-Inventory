from typing import Literal

from fastapi import HTTPException
from sqlalchemy import Select, func, or_, select
from sqlalchemy.sql.functions import count
from sqlalchemy.orm import Session

from backend.models import Inventory
from backend.schemas.inventory import BulkPriceUpdate, PriceUpdate, QuantityUpdate

StockStatus = Literal["", "in-stock", "low-stock", "out-of-stock"]
SortField = Literal["name", "price", "stock", "value"]
SortOrder = Literal["asc", "desc"]


def _filters(
    search: str,
    category: str,
    supplier: str,
    stock_status: StockStatus,
) -> list:
    filters = []
    if search:
        term = f"%{search}%"
        filters.append(
            or_(
                Inventory.item_name.ilike(term),
                Inventory.category.ilike(term),
                Inventory.supplier.ilike(term),
            )
        )
    if category:
        filters.append(Inventory.category == category)
    if supplier:
        filters.append(Inventory.supplier == supplier)
    if stock_status == "out-of-stock":
        filters.append(Inventory.stock_quantity == 0)
    elif stock_status == "low-stock":
        filters.append(Inventory.stock_quantity.between(1, 49))
    elif stock_status == "in-stock":
        filters.append(Inventory.stock_quantity >= 50)
    return filters


def list_inventory(
    database: Session,
    *,
    search: str,
    category: str,
    supplier: str,
    stock_status: StockStatus,
    sort: SortField,
    order: SortOrder,
    page: int,
    page_size: int,
) -> dict:
    filters = _filters(search, category, supplier, stock_status)
    sort_columns = {
        "name": func.lower(Inventory.item_name),
        "price": Inventory.price,
        "stock": Inventory.stock_quantity,
        "value": Inventory.price * Inventory.stock_quantity,
    }
    sort_column = sort_columns[sort]
    order_by = sort_column.desc() if order == "desc" else sort_column.asc()
    query: Select[tuple[Inventory]] = select(Inventory).where(*filters).order_by(
        order_by, Inventory.item_id.asc()
    )
    items = database.scalars(query.offset((page - 1) * page_size).limit(page_size)).all()
    total = database.scalar(select(count()).select_from(Inventory).where(*filters)) or 0
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        **inventory_options(database),
    }


def inventory_options(database: Session) -> dict:
    categories = database.scalars(
        select(Inventory.category).distinct().order_by(func.lower(Inventory.category))
    ).all()
    suppliers = database.scalars(
        select(Inventory.supplier).distinct().order_by(func.lower(Inventory.supplier))
    ).all()
    return {
        "categories": categories,
        "suppliers": suppliers,
    }


def update_quantity(
    database: Session, item_id: int, update: QuantityUpdate, *, commit: bool = True
) -> Inventory:
    item = database.get(Inventory, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Inventory item not found")
    item.stock_quantity = update.stock_quantity
    if commit:
        database.commit()
        database.refresh(item)
    else:
        database.flush()
    return item


def update_price(
    database: Session, item_id: int, update: PriceUpdate, *, commit: bool = True,
) -> Inventory:
    item = database.get(Inventory, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Inventory item not found")
    item.price = update.price
    if commit:
        database.commit()
        database.refresh(item)
    else:
        database.flush()
    return item


def bulk_price_update(database: Session, update: BulkPriceUpdate) -> dict:
    items = database.scalars(
        select(Inventory).where(Inventory.item_id.in_(update.item_ids))
    ).all()
    if len(items) != len(update.item_ids):
        raise HTTPException(
            status_code=404, detail="One or more inventory items were not found"
        )
    for item in items:
        if item.price is None:
            raise HTTPException(
                status_code=400,
                detail=f"Inventory item {item.item_id} does not have a price",
            )
        if update.adjustment_type == "percent":
            new_price = item.price * (1 + update.adjustment / 100)
        else:
            new_price = item.price + update.adjustment
        if new_price < 0:
            raise HTTPException(
                status_code=400,
                detail="Price adjustment cannot result in a negative price",
            )
        item.price = round(new_price, 2)
    database.commit()
    return {"updated": len(items), "item_ids": update.item_ids}
