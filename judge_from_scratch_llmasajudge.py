from llmasajudge import LLMAsAJudge
from datasets import load_dataset
import weave

weave.init("judge_from_scratch_llmasajudge")

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


# Custom parser for the decision
def decision_parser(response: str) -> bool:
    """Parse the judge response to return True if CORRECT."""
    response = response.strip().upper()
    return "CORRECT" in response and "INCORRECT" not in response


# Initialize LLMAsAJudge with single model and custom parser
judge = LLMAsAJudge(
    models=["gpt-4o"],
    use_fully_custom_prompt=True,
    output_parser=decision_parser,
    verbose=True,
    litellm_cache_dir='./judge_litellm_cache'
)

# Get judgment
result = judge.judge(prompt=prompt)

# Extract decision from result
# LLMAsAJudge returns {"correct": bool, "mode": str, "votes": list}
# The 'correct' key contains the parsed boolean result
is_correct = result.get("correct", False)

print("Judge returned:", "CORRECT" if is_correct else "INCORRECT")
print("Full result:", result)

assert is_correct, f"Expected judge to say CORRECT, but got: {result}"
