
import openai
from datasets import load_dataset
import weave; weave.init("judge_from_scratch")
# Load one QA sample from SimpleQA
data = load_dataset("basicv8vc/SimpleQA", split="test")
sample = data[0]
question = sample["problem"]
gold_answer = sample["answer"]
predicted_answer = gold_answer  # Using gold as prediction

# Comparator-style prompt for strict judgment
prompt = f"""
You are a strict evaluator for factual QA.
Given a question, a gold answer, and a predicted answer, is the predicted answer factually correct with respect to the gold answer?
Question: {question}
Gold answer: {gold_answer}
Predicted answer: {predicted_answer}
Respond only with CORRECT, INCORRECT, or NOT_ATTEMPTED.
"""

# Judge call with deterministic sampling
response = openai.chat.completions.create(
    model="gpt-4o",
    temperature=0,
    messages=[{"role": "user", "content": prompt}]
)

decision = response.choices[0].message.content.strip().upper()


print("Judge returned:", decision)
assert "CORRECT" in decision, f"Expected judge to say CORRECT, but got: {decision}"

