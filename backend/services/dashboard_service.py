from sqlalchemy import func, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.functions import count

from backend.models import Inventory


def dashboard(database: Session) -> dict:
    totals = database.execute(
        select(
            count(Inventory.item_id).label("items"),
            func.coalesce(func.sum(Inventory.stock_quantity), 0).label("units"),
            func.coalesce(
                func.sum(Inventory.price * Inventory.stock_quantity), 0
            ).label("value"),
            func.coalesce(func.avg(Inventory.price), 0).label("average_price"),
        )
    ).one()
    low_stock = database.scalar(
        select(count()).select_from(Inventory).where(
            Inventory.stock_quantity.between(1, 49)
        )
    ) or 0
    out_of_stock = database.scalar(
        select(count()).select_from(Inventory).where(Inventory.stock_quantity == 0)
    ) or 0
    data_issues = database.scalar(
        select(count()).select_from(Inventory).where(Inventory.stock_quantity < 0)
    ) or 0
    attention_count = database.scalar(
        select(count()).select_from(Inventory).where(Inventory.stock_quantity < 50)
    ) or 0
    unpriced_items = database.scalar(
        select(count()).select_from(Inventory).where(Inventory.price.is_(None))
    ) or 0
    categories = database.execute(
        select(
            Inventory.category,
            count(Inventory.item_id).label("item_count"),
            func.sum(Inventory.stock_quantity).label("units"),
            func.sum(Inventory.price * Inventory.stock_quantity).label("value"),
        )
        .group_by(Inventory.category)
        .order_by(func.sum(Inventory.price * Inventory.stock_quantity).desc())
        .limit(8)
    ).all()
    suppliers = database.execute(
        select(
            Inventory.supplier,
            count(Inventory.item_id).label("item_count"),
            func.sum(Inventory.stock_quantity).label("units"),
            func.sum(Inventory.price * Inventory.stock_quantity).label("value"),
        )
        .group_by(Inventory.supplier)
        .order_by(func.sum(Inventory.price * Inventory.stock_quantity).desc())
    ).all()
    attention = database.execute(
        select(
            Inventory.item_id,
            Inventory.item_name,
            Inventory.stock_quantity,
            Inventory.price,
            Inventory.supplier,
        )
        .where(Inventory.stock_quantity < 50)
        .order_by(Inventory.stock_quantity.asc(), func.lower(Inventory.item_name))
        .limit(6)
    ).all()
    return {
        "totals": {
            **totals._asdict(),
            "low_stock": low_stock,
            "out_of_stock": out_of_stock,
            "data_issues": data_issues,
            "attention_count": attention_count,
            "unpriced_items": unpriced_items,
        },
        "categories": [row._asdict() for row in categories],
        "suppliers": [row._asdict() for row in suppliers],
        "attention": [row._asdict() for row in attention],
    }
