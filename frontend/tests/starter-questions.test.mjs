import assert from "node:assert/strict";
import test from "node:test";
import { chooseStarterQuestions, starterQuestions } from "../src/starter-questions.ts";

test("contains exactly 30 unique project prompts, each at most ten words", () => {
  assert.equal(starterQuestions.length, 30);
  assert.equal(new Set(starterQuestions.map(item => item.question)).size, 30);
  for (const item of starterQuestions) {
    assert.ok(item.question.trim().split(/\s+/).length <= 10, item.question);
    assert.ok(["sql", "faq", "quantity"].includes(item.agent));
  }
});

test("chooses three distinct prompts, one per agent, without changing the bank", () => {
  const before = JSON.stringify(starterQuestions);
  for (let attempt = 0; attempt < 100; attempt++) {
    const selected = chooseStarterQuestions();
    assert.equal(selected.length, 3);
    assert.equal(new Set(selected.map(item => item.question)).size, 3);
    assert.deepEqual(selected.map(item => item.agent), ["sql", "faq", "quantity"]);
    assert.ok(selected.every(item => starterQuestions.includes(item)));
  }
  assert.equal(JSON.stringify(starterQuestions), before);
});

test("random selection can reach every question in the bank", () => {
  const original = Math.random;
  const seen = new Set();
  try {
    for (let index = 0; index < 10; index++) {
      Math.random = () => (index + 0.5) / 10;
      for (const item of chooseStarterQuestions()) seen.add(item.question);
    }
  } finally {
    Math.random = original;
  }
  assert.equal(seen.size, 30);
});
