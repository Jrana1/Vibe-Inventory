import assert from "node:assert/strict";
import test from "node:test";
import { agentNames, snapshotValue } from "../src/ai-types.ts";

test("price previews distinguish missing, zero and numeric prices", () => {
  const item = { item_id: 379, item_name: "Example item" };
  assert.equal(snapshotValue({ ...item, price: null }), "Unpriced");
  assert.equal(snapshotValue({ ...item, price: 0 }), "$0.00");
  assert.equal(snapshotValue({ ...item, price: 120 }), "$120.00");
  assert.equal(snapshotValue({ ...item, price: 19.99 }), "$19.99");
  assert.equal(snapshotValue({ ...item, stock_quantity: 3 }), "3");
  assert.equal(agentNames.price, "Price setting");
});
