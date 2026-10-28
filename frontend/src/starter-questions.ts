import type { SuggestedQuestion } from "./ai-types";

export const starterQuestions: readonly SuggestedQuestion[] = [
  { agent: "sql", question: "Which products have the highest unit prices?" },
  { agent: "sql", question: "Show all unique inventory suppliers." },
  { agent: "sql", question: "Which products are out of stock?" },
  { agent: "sql", question: "Which products have low stock?" },
  { agent: "sql", question: "Show inventory value by supplier." },
  { agent: "sql", question: "How many products belong to each category?" },
  { agent: "sql", question: "Which products have missing prices?" },
  { agent: "sql", question: "Show products with negative stock quantities." },
  { agent: "sql", question: "Which supplier provides the most products?" },
  { agent: "sql", question: "Show total stock units by category." },
  { agent: "faq", question: "How does the manual handle negative stock?" },
  { agent: "faq", question: "What does the manual say about stock counts?" },
  { agent: "faq", question: "How should stock discrepancies be handled?" },
  { agent: "faq", question: "What supplier controls does the procurement policy describe?" },
  { agent: "faq", question: "What does the procurement policy require before ordering?" },
  { agent: "faq", question: "How does the manual document inventory adjustments?" },
  { agent: "faq", question: "What issues appear in the monthly audit report?" },
  { agent: "faq", question: "What recommendations appear in the monthly audit report?" },
  { agent: "faq", question: "What does the audit report say about discrepancies?" },
  { agent: "faq", question: "Who approves stock adjustments according to the documents?" },
  { agent: "quantity", question: "Preview setting item 1 quantity to 10." },
  { agent: "quantity", question: "Preview setting item 2 quantity to 20." },
  { agent: "quantity", question: "Preview setting item 3 quantity to 15." },
  { agent: "quantity", question: "Preview setting item 4 quantity to 50." },
  { agent: "quantity", question: "Preview setting item 5 quantity to zero." },
  { agent: "quantity", question: "Set item 1 to 10; item 2 to 20." },
  { agent: "quantity", question: "Set item 3 to 15; item 4 to 25." },
  { agent: "quantity", question: "Preview setting items 1 and 2 quantities to 30." },
  { agent: "quantity", question: "Preview setting items 3 and 4 quantities to 40." },
  { agent: "quantity", question: "Preview setting items 4 and 5 quantities to zero." },
];

export function chooseStarterQuestions(): SuggestedQuestion[] {
  return (["sql", "faq", "quantity"] as const).map(agent => {
    const questions = starterQuestions.filter(question => question.agent === agent);
    return questions[Math.floor(Math.random() * questions.length)];
  });
}
