import argparse
import random
from typing import Dict, List
from datasets import load_dataset
from llmasajudge import LLMAsAJudge
import weave

# Judging criteria for comparing two chatbot responses
ARENA_JUDGE_PROMPT = """
You are an expert evaluator comparing two AI assistant responses to the same user question.

Evaluate the responses based on these criteria:
1. **Helpfulness**: Does the response directly answer the question?
2. **Accuracy**: Is the information factually correct?
3. **Completeness**: Does it cover all important aspects?
4. **Clarity**: Is it well-organized and easy to understand?
5. **Conciseness**: Does it avoid unnecessary verbosity?

Question: <<QUESTION>>

Response A:
<<RESPONSE_A>>

Response B:
<<RESPONSE_B>>

Which response is better overall? Consider all criteria above.
Return ONLY one of:
A: Response A is better
B: Response B is better
C: They are roughly equal (tie)

Just return A, B, or C.
""".strip()


def arena_judge_parser(response: str) -> str:
    """Parse judge response to A, B, or C."""
    if response is None:
        return "tie"
    txt = str(response).strip().upper()
    if "A" in txt and "B" not in txt:
        return "model_a"
    elif "B" in txt and "A" not in txt:
        return "model_b"
    else:
        return "tie"


class ArenaDataset:
    def __init__(self, seed: int = 42, judge_models: str = "gpt-4o"):
        self.rng = random.Random(seed)
        print("Loading Chatbot Arena dataset...")
        self.dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")
        print(f"Loaded {len(self.dataset)} conversations")

        # Parse comma-separated models and strip litellm/ prefix
        models = [m.strip().replace("litellm/", "") for m in judge_models.split(",")]
        print(f"Using judge models: {models}")

        self.judge = LLMAsAJudge(
            models=models,
            use_fully_custom_prompt=True,
            output_parser=arena_judge_parser,
            verbose=False,
            litellm_cache_dir='./judge_litellm_cache'
        )

    def get_examples(self, num_samples: int = 10) -> List[Dict]:
        """Get random examples from the dataset."""
        n_total = len(self.dataset)
        n = min(num_samples, n_total)
        idx = list(range(n_total))
        self.rng.shuffle(idx)
        idx = idx[:n]

        out = []
        for i in idx:
            row = self.dataset[i]

            # Get the user's question (first turn)
            conversation_a = row.get("conversation_a", [])
            conversation_b = row.get("conversation_b", [])

            if not conversation_a or not conversation_b:
                continue

            # Extract question and responses
            question = conversation_a[0].get("content", "") if conversation_a else ""
            response_a = conversation_a[1].get("content", "") if len(conversation_a) > 1 else ""
            response_b = conversation_b[1].get("content", "") if len(conversation_b) > 1 else ""

            if not question or not response_a or not response_b:
                continue

            # Get ground truth winner from dataset
            winner = row.get("winner", "tie")  # Can be "model_a", "model_b", or "tie"
            if "tie" in winner.lower():
                winner = "tie" # catch weird cases with explanations

            # Create judging prompt
            judge_prompt = (
                ARENA_JUDGE_PROMPT
                .replace("<<QUESTION>>", question)
                .replace("<<RESPONSE_A>>", response_a)
                .replace("<<RESPONSE_B>>", response_b)
            )

            out.append({
                "question_id": row.get("question_id", ""),
                "question": question,
                "response_a": response_a,
                "response_b": response_b,
                "model_a": row.get("model_a", ""),
                "model_b": row.get("model_b", ""),
                "ground_truth_winner": winner,
                "judge_prompt": judge_prompt,
                "metadata": {
                    "index": i,
                    "language": row.get("language", ""),
                    "turn": row.get("turn", 1),
                }
            })

        return out

    @weave.op
    def get_judgment(self, judge_prompt: str) -> str:
        """Get judgment from LLMAsAJudge."""
        try:
            result = self.judge.judge(prompt=judge_prompt)
            # LLMAsAJudge returns {"correct": bool, "mode": str, "votes": list}
            # We use custom parser so mode should be "model_a", "model_b", or "tie"
            return result.get("result", "tie")
        except Exception as e:
            print(f"Judge error: {e}")
            return "tie"


def run_eval(num_samples: int = 10, seed: int = 42, weave_project: str = "arena", judge_models: str = "gpt-4o"):
    """Run Arena evaluation."""
    # Initialize Weave
    weave.init(weave_project)

    print(f"Running Arena eval with num_samples={num_samples}, judge_models={judge_models}")

    # Create evaluation logger
    ev = weave.EvaluationLogger(
        name=f"arena_judge_eval_{judge_models.replace('/', '_').replace(',', '_')}",
        model={"name": judge_models},
        scorers=["agreement"],
    )

    random.seed(seed)
    ds = ArenaDataset(seed=seed, judge_models=judge_models)
    examples = ds.get_examples(num_samples=num_samples)

    print(f"\nEvaluating {len(examples)} examples...")

    for ex in examples:
        print(f"\n{'='*80}")
        print(f"Q: {ex['question'][:100]}...")
        print(f"Model A ({ex['model_a']}): {ex['response_a'][:100]}...")
        print(f"Model B ({ex['model_b']}): {ex['response_b'][:100]}...")
        print(f"Ground Truth Winner: {ex['ground_truth_winner']}")

        # Get judge's decision
        judge_decision = ds.get_judgment(ex["judge_prompt"])
        print(f"Judge Decision: {judge_decision}")

        # Check if judge agrees with ground truth
        agreement = 1.0 if judge_decision == ex["ground_truth_winner"] else 0.0
        print(f"Agreement: {agreement}")

        # Log to Weave
        ev.log_example(
            inputs={
                "question": ex["question"],
                "response_a": ex["response_a"],
                "response_b": ex["response_b"],
                "model_a": ex["model_a"],
                "model_b": ex["model_b"],
                "ground_truth_winner": ex["ground_truth_winner"],
            },
            output={
                "judge_decision": judge_decision,
            },
            scores={
                "agreement": agreement
            }
        )

    # Log summary
    ev.log_summary()
    print(f"\n{'='*80}")
    print(f"Evaluation complete! View at: {ev.ui_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--weave_project", type=str, default="arena", help="Weave project name")
    parser.add_argument("--judge_models", type=str, default="gpt-4o", help="Comma-separated judge models (e.g., 'litellm/openai/gpt-4o,litellm/openai/gpt-5')")
    args = parser.parse_args()

    run_eval(args.num_samples, args.seed, args.weave_project, args.judge_models)
