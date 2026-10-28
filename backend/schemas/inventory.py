from typing import Literal

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class InventoryItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    item_id: int
    category: str
    item_name: str
    price: float | None
    stock_quantity: int
    supplier: str


class InventoryListResponse(BaseModel):
    items: list[InventoryItemResponse]
    total: int
    page: int
    page_size: int
    categories: list[str]
    suppliers: list[str]


class QuantityUpdate(BaseModel):
    stock_quantity: int = Field(ge=0)


class PriceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    price: float = Field(ge=0, le=1_000_000, strict=True)

    @field_validator("price")
    @classmethod
    def cents_only(cls, value: float) -> float:
        amount = Decimal(str(value))
        if amount != amount.quantize(Decimal("0.01")):
            raise ValueError("Price must have at most two decimal places")
        return value


class BulkPriceUpdate(BaseModel):
    item_ids: list[int] = Field(min_length=1)
    adjustment_type: Literal["percent", "fixed"]
    adjustment: float

    @model_validator(mode="after")
    def validate_adjustment(self) -> "BulkPriceUpdate":
        if self.adjustment_type == "percent" and self.adjustment < -100:
            raise ValueError("Percentage adjustment cannot reduce prices below zero")
        if len(self.item_ids) != len(set(self.item_ids)):
            raise ValueError("item_ids must not contain duplicates")
        return self


class BulkPriceUpdateResponse(BaseModel):
    updated: int
    item_ids: list[int]
