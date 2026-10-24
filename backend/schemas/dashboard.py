from pydantic import BaseModel


class DashboardTotals(BaseModel):
    items: int
    units: int
    value: float
    average_price: float
    low_stock: int
    out_of_stock: int
    data_issues: int
    attention_count: int
    unpriced_items: int


class CategorySummary(BaseModel):
    category: str
    item_count: int
    units: int
    value: float


class SupplierSummary(BaseModel):
    supplier: str
    item_count: int
    units: int
    value: float


class AttentionItem(BaseModel):
    item_id: int
    item_name: str
    stock_quantity: int
    price: float | None
    supplier: str


class DashboardResponse(BaseModel):
    totals: DashboardTotals
    categories: list[CategorySummary]
    suppliers: list[SupplierSummary]
    attention: list[AttentionItem]
