import json
import time
from uuid import uuid4

from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import Float, String, Text, func, select, text
from sqlalchemy.orm import Mapped, Session, mapped_column

from backend.core.database import Base
from backend.models import Inventory
from backend.schemas.ai import InventoryAction, PriceAction
from backend.schemas.inventory import PriceUpdate, QuantityUpdate
from backend.services import inventory_service

action_adapter = TypeAdapter(InventoryAction)


class AgentRun(Base):
    __tablename__ = "agent_runs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String)
    result: Mapped[str] = mapped_column(Text)


class AgentProposal(Base):
    __tablename__ = "agent_proposals"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    created_at: Mapped[float] = mapped_column(Float)
    expires_at: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String)
    action: Mapped[str] = mapped_column(Text)
    before: Mapped[str] = mapped_column(Text)
    after: Mapped[str] = mapped_column(Text)
    decided_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String, nullable=True)


def recover_interrupted_runs(database: Session) -> None:
    for record in database.scalars(select(AgentRun).where(AgentRun.status == "running")):
        result = json.loads(record.result)
        result["status"] = "failed"
        result["answer"] = "Run interrupted by backend restart. No inventory changes were applied; submit a fresh request."
        record.status = "failed"
        record.result = json.dumps(result)
    for record in database.scalars(select(AgentRun).where(
        AgentRun.status == "completed",
        func.json_extract(AgentRun.result, "$.explanation_status") == "streaming",
    )):
        result = json.loads(record.result)
        result["explanation_status"] = "interrupted"
        result["explanation_error"] = "Explanation interrupted by backend restart; validated results remain available."
        record.result = json.dumps(result)
    for proposal in database.scalars(select(AgentProposal).where(AgentProposal.status == "pending")):
        try:
            action_adapter.validate_json(proposal.action)
        except ValidationError:
            proposal.status = "rejected"
            proposal.decided_at = time.time()
    database.commit()


def item_snapshot(database: Session, action: InventoryAction) -> list[dict]:
    ids = sorted(update.item_id for update in action.updates)
    items = database.scalars(
        select(Inventory).where(Inventory.item_id.in_(ids)).order_by(Inventory.item_id)
    ).all()
    if len(items) != len(ids):
        raise HTTPException(404, "One or more inventory items were not found")
    return [
        {"item_id": item.item_id, "item_name": item.item_name,
         **({"price": item.price} if isinstance(action, PriceAction)
            else {"stock_quantity": item.stock_quantity})}
        for item in items
    ]


def proposal_view(proposal: AgentProposal) -> dict:
    return {
        "id": proposal.id, "run_id": proposal.run_id, "status": proposal.status,
        "created_at": proposal.created_at, "expires_at": proposal.expires_at,
        "action": json.loads(proposal.action), "before": json.loads(proposal.before),
        "after": json.loads(proposal.after), "decided_at": proposal.decided_at,
        "trace_id": proposal.trace_id,
    }


def create_proposal(database: Session, run_id: str, action: InventoryAction) -> dict:
    before = item_snapshot(database, action)
    after = [dict(row) for row in before]
    values = (
        {update.item_id: update.price for update in action.updates}
        if isinstance(action, PriceAction)
        else {update.item_id: update.stock_quantity for update in action.updates}
    )
    field = "price" if isinstance(action, PriceAction) else "stock_quantity"
    for row in after:
        row[field] = values[row["item_id"]]
    now = time.time()
    proposal = AgentProposal(
        id=str(uuid4()), run_id=run_id, created_at=now, expires_at=now + 600,
        status="pending", action=json.dumps(action.model_dump()), before=json.dumps(before),
        after=json.dumps(after),
    )
    database.add(proposal)
    database.flush()
    return proposal_view(proposal)


def decide(database: Session, proposal_id: str, approve: bool, trace_id: str | None) -> dict:
    # Serialize decisions and inventory edits; approval status and changes commit together.
    database.execute(text("BEGIN IMMEDIATE"))
    proposal = database.get(AgentProposal, proposal_id)
    if proposal is None:
        raise HTTPException(404, "Proposal not found")
    try:
        action = action_adapter.validate_json(proposal.action)
    except ValidationError as exc:
        raise HTTPException(409, "This proposal belongs to the removed AI workflow; submit a new request") from exc
    desired = "approved" if approve else "rejected"
    if proposal.status == desired:
        return proposal_view(proposal)
    if proposal.status != "pending":
        raise HTTPException(409, f"Proposal is already {proposal.status}")
    if approve and proposal.expires_at <= time.time():
        raise HTTPException(409, "Proposal expired; submit a fresh request")
    if approve:
        if item_snapshot(database, action) != json.loads(proposal.before):
            raise HTTPException(409, "Inventory changed since this proposal; submit a fresh request")
        if isinstance(action, PriceAction):
            for update in action.updates:
                inventory_service.update_price(
                    database, update.item_id, PriceUpdate(price=update.price), commit=False,
                )
        else:
            for update in action.updates:
                inventory_service.update_quantity(
                    database, update.item_id,
                    QuantityUpdate(stock_quantity=update.stock_quantity), commit=False,
                )
    proposal.status = desired
    proposal.decided_at = time.time()
    proposal.trace_id = trace_id
    database.commit()
    return proposal_view(proposal)
