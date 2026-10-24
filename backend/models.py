from sqlalchemy import Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class Inventory(Base):
    __tablename__ = "inventory"

    item_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category: Mapped[str] = mapped_column(String)
    item_name: Mapped[str] = mapped_column(String)
    price: Mapped[float | None] = mapped_column(Float)
    stock_quantity: Mapped[int] = mapped_column(Integer)
    supplier: Mapped[str] = mapped_column(String)
