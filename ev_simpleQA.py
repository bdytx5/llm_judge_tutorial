import re
import random
from typing import Dict, List, Optional
from datasets import load_dataset
from llmasajudge import LLMAsAJudge
import weave

SIMPLEQA_GRADING_PROMPT = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.

The following are examples of CORRECT predicted answers.
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer: sasha and malia obama

The following are examples of INCORRECT predicted answers.
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer: Barack Obama does not have any children.

The following are examples of NOT_ATTEMPTED predicted answers.
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer: I don't know.

Here is a new example.
Question: <<QUESTION>>
Gold target: <<TARGET>>
Predicted answer: <<PREDICTED_ANSWER>>

Grade the predicted answer as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED
Just return A, B, or C.
""".strip()


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def simpleqa_bool_parser(response: str) -> bool:
    if response is None:
        return False
    txt = str(response).strip().upper()
    if "B" in txt or "C" in txt:
        return False
    elif "A" in txt:
        return True
    else:
        return False


class SimpleQADataset:
    def __init__(self, seed: int = 42, judge_models: str = "gpt-4o"):
        self.rng = random.Random(seed)
        self.dataset = load_dataset("basicv8vc/SimpleQA", split="test")

        # Parse comma-separated models and strip litellm/ prefix
        models = [m.strip().replace("litellm/", "") for m in judge_models.split(",")]
        print(f"Using judge models: {models}")

        self.judge = LLMAsAJudge(
            models=models,
            use_fully_custom_prompt=True,
            output_parser=simpleqa_bool_parser,
            verbose=False,
            litellm_cache_dir='./judge_litellm_cache'
        )

    def get_examples(self, num_samples: Optional[int] = None) -> List[Dict]:
        n_total = len(self.dataset)
        n = n_total if num_samples is None else min(num_samples, n_total)
        idx = list(range(n_total))
        self.rng.shuffle(idx)
        idx = idx[:n]

        out = []
        for i in idx:
            row = self.dataset[i]
            q = _clean(row.get("problem"))
            a = _clean(row.get("answer"))

            grading_prompt = (
                SIMPLEQA_GRADING_PROMPT
                .replace("<<QUESTION>>", q)
                .replace("<<TARGET>>", a)
                .replace("<<PREDICTED_ANSWER>>", "<RESPONSE>")
            )

            out.append({
                "question": q,
                "answer": a,
                "grading_prompt": grading_prompt,
                "metadata": {"index": i}
            })
        return out

    @weave.op
    def get_score(self, generated_answer: str, grading_prompt: str, metadata: Dict) -> bool:
        if not isinstance(generated_answer, str):
            generated_answer = str(generated_answer)
        prompt = grading_prompt.replace("<RESPONSE>", generated_answer)
        try:
            score = self.judge.judge(prompt=prompt)
            # LLMAsAJudge returns a dict: {"correct": bool, "mode": str, "votes": list}
            return score.get("correct", False)
        except Exception as e:
            print(f"judge error: {e}")
            return False


import argparse
from litellm import completion


@weave.op
def generate_answer(question: str, model: str, temperature: float) -> str:
    """Generate an answer using LiteLLM."""
    messages = [{"role": "user", "content": question}]

    # Strip litellm/ prefix if present
    model_name = model.replace("litellm/", "")

    # Don't pass temperature for gpt-5 models
    kwargs = {}
    if "gpt-5" not in model_name.lower():
        kwargs["temperature"] = temperature

    response = completion(model=model_name, messages=messages, **kwargs)
    return response.choices[0].message.content.strip()


def run_eval(model: str, num_samples: int = 10, seed: int = 42, judge_models: str = "gpt-4o"):
    """Run SimpleQA evaluation with the given model."""
    # Initialize Weave
    weave.init("byyoung3/simpleqa")

    # Set temperature based on model
    temperature = 1.0 if "gpt-5" in model.lower() else 0.0

    print(f"Running SimpleQA eval with model={model}, judge_models={judge_models}, num_samples={num_samples}, temp={temperature}")

    # Create evaluation logger
    ev = weave.EvaluationLogger(
        name=f"simpleqa_{model.replace('/', '_')}_{judge_models.replace('/', '_').replace(',', '_')}",
        model={"name": model},
        scorers=["correct"],
    )

    random.seed(seed)
    ds = SimpleQADataset(seed=seed, judge_models=judge_models)
    examples = ds.get_examples(num_samples=num_samples)

    for ex in examples:
        print(f"\nQ: {ex['question']}")
        print(f"Gold: {ex['answer']}")

        # Generate answer using the model
        generated_answer = generate_answer(ex["question"], model, temperature)
        print(f"Generated: {generated_answer}")

        # Score the answer
        score = ds.get_score(generated_answer, ex["grading_prompt"], ex["metadata"])
        print(f"Correct: {score}")

        # Log to Weave
        ev.log_example(
            inputs={"question": ex["question"], "gold_answer": ex["answer"]},
            output={"generated_answer": generated_answer},
            scores={"correct": 1.0 if score else 0.0}
        )

    # Log summary
    ev.log_summary()
    print(f"\nEvaluation complete! View at: {ev.ui_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="LiteLLM model ID (e.g., litellm/openai/gpt-5-nano)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--judge_models", type=str, default="gpt-4o", help="Comma-separated judge models (e.g., 'litellm/openai/gpt-4o,litellm/openai/gpt-5')")
    args = parser.parse_args()

    run_eval(args.model, args.num_samples, args.seed, args.judge_models)