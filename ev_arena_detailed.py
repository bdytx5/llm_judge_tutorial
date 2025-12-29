import argparse
import json
import random
from typing import Dict, List
from datasets import load_dataset
from llmasajudge import LLMAsAJudge
import weave

# Detailed scoring prompt for a SINGLE response
DETAILED_SCORE_PROMPT = """
You are an expert evaluator assessing an AI assistant's response to a user question.

Rate the response on the following criteria using a scale of 1-5 (1=Poor, 5=Excellent):

1. **Helpfulness**: Does the response directly answer the question?
2. **Accuracy**: Is the information factually correct?
3. **Completeness**: Does it cover all important aspects?
4. **Clarity**: Is it well-organized and easy to understand?
5. **Conciseness**: Does it avoid unnecessary verbosity?

Question: <<QUESTION>>

Response:
<<RESPONSE>>

Please respond in the following JSON format:
{
  "helpfulness": <score 1-5>,
  "accuracy": <score 1-5>,
  "completeness": <score 1-5>,
  "clarity": <score 1-5>,
  "conciseness": <score 1-5>
}

Return ONLY the JSON, no other text.
""".strip()


def parse_score_json(response: str) -> Dict[str, float]:
    """Parse JSON response with scores."""
    print(f"DEBUG parse_score_json received: {response}")
    print(f"DEBUG type: {type(response)}")
    try:
        # Extract JSON from response
        response = response.strip()
        # Find JSON object in the response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            json_str = response[start:end]
            print(f"DEBUG extracted JSON: {json_str}")
            scores = json.loads(json_str)
            print(f"DEBUG parsed scores: {scores}")
            # Ensure all scores are present and convert to float
            return {
                "helpfulness": float(scores.get("helpfulness", 3)),
                "accuracy": float(scores.get("accuracy", 3)),
                "completeness": float(scores.get("completeness", 3)),
                "clarity": float(scores.get("clarity", 3)),
                "conciseness": float(scores.get("conciseness", 3)),
            }
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Response: {response}")

    # Return default scores if parsing fails
    return {
        "helpfulness": 3.0,
        "accuracy": 3.0,
        "completeness": 3.0,
        "clarity": 3.0,
        "conciseness": 3.0,
    }


def calculate_average(scores: Dict[str, float]) -> float:
    """Calculate average score across all criteria."""
    return sum(scores.values()) / len(scores)


class ArenaDetailedDataset:
    def __init__(self, seed: int = 42, judge_models: str = "gpt-4o"):
        self.rng = random.Random(seed)
        self.judge_models = judge_models
        print("Loading Chatbot Arena dataset...")
        self.dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")
        print(f"Loaded {len(self.dataset)} conversations")

        # Parse comma-separated models and strip litellm/ prefix
        models = [m.strip().replace("litellm/", "") for m in judge_models.split(",")]
        print(f"Using judge models: {models}")

        # Initialize LLMAsAJudge
        self.judge = LLMAsAJudge(
            models=models,
            use_fully_custom_prompt=True,
            output_parser=parse_score_json,
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

            out.append({
                "question_id": row.get("question_id", ""),
                "question": question,
                "response_a": response_a,
                "response_b": response_b,
                "model_a": row.get("model_a", ""),
                "model_b": row.get("model_b", ""),
                "ground_truth_winner": winner,
                "metadata": {
                    "index": i,
                    "language": row.get("language", ""),
                    "turn": row.get("turn", 1),
                }
            })

        return out

    @weave.op
    def score_response(self, question: str, response: str) -> Dict[str, float]:
        """Score a single response on multiple criteria."""
        prompt = (
            DETAILED_SCORE_PROMPT
            .replace("<<QUESTION>>", question)
            .replace("<<RESPONSE>>", response)
        )

        try:
            result = self.judge.judge(prompt=prompt)
            # With updated llmasajudge, scores are in result['scores']
            if 'scores' in result and isinstance(result['scores'], dict):
                return result['scores']
            else:
                print(f"Unexpected result format: {result}")
                return {
                    "helpfulness": 3.0,
                    "accuracy": 3.0,
                    "completeness": 3.0,
                    "clarity": 3.0,
                    "conciseness": 3.0,
                }
        except Exception as e:
            print(f"Scoring error: {e}")
            return {
                "helpfulness": 3.0,
                "accuracy": 3.0,
                "completeness": 3.0,
                "clarity": 3.0,
                "conciseness": 3.0,
            }


def run_eval(
    num_samples: int = 10,
    seed: int = 42,
    weave_project: str = "arena-detailed",
    judge_models: str = "gpt-4o",
    supervised_comparison: bool = True
):
    """Run detailed Arena evaluation."""
    # Initialize Weave
    weave.init(weave_project)

    print(f"Running Arena detailed eval with num_samples={num_samples}, judge_models={judge_models}")
    print(f"Supervised comparison: {supervised_comparison}")

    # Define scorers based on mode
    if supervised_comparison:
        scorers = [
            # Response A scores
            "response_a_helpfulness",
            "response_a_accuracy",
            "response_a_completeness",
            "response_a_clarity",
            "response_a_conciseness",
            "response_a_avg",
            # Response B scores
            "response_b_helpfulness",
            "response_b_accuracy",
            "response_b_completeness",
            "response_b_clarity",
            "response_b_conciseness",
            "response_b_avg",
            # Comparison
            "predicted_winner",
            "agreement",
        ]
    else:
        scorers = [
            # Response A scores
            "response_a_helpfulness",
            "response_a_accuracy",
            "response_a_completeness",
            "response_a_clarity",
            "response_a_conciseness",
            "response_a_avg",
            # Response B scores
            "response_b_helpfulness",
            "response_b_accuracy",
            "response_b_completeness",
            "response_b_clarity",
            "response_b_conciseness",
            "response_b_avg",
        ]

    # Create evaluation logger
    ev = weave.EvaluationLogger(
        name=f"arena_detailed_{judge_models.replace('/', '_').replace(',', '_')}",
        model={"name": judge_models},
        scorers=scorers,
    )

    random.seed(seed)
    ds = ArenaDetailedDataset(seed=seed, judge_models=judge_models)
    examples = ds.get_examples(num_samples=num_samples)

    print(f"\nEvaluating {len(examples)} examples...")

    for ex in examples:
        print(f"\n{'='*80}")
        print(f"Q: {ex['question'][:100]}...")
        print(f"Model A ({ex['model_a']})")
        print(f"Model B ({ex['model_b']})")
        print(f"Ground Truth Winner: {ex['ground_truth_winner']}")

        # Score response A
        print("\nScoring Response A...")
        scores_a = ds.score_response(ex["question"], ex["response_a"])
        avg_a = calculate_average(scores_a)
        print(f"Response A scores: {scores_a}")
        print(f"Response A average: {avg_a:.2f}")

        # Score response B
        print("\nScoring Response B...")
        scores_b = ds.score_response(ex["question"], ex["response_b"])
        avg_b = calculate_average(scores_b)
        print(f"Response B scores: {scores_b}")
        print(f"Response B average: {avg_b:.2f}")

        # Build scores dict for logging
        log_scores = {
            # Response A
            "response_a_helpfulness": scores_a["helpfulness"],
            "response_a_accuracy": scores_a["accuracy"],
            "response_a_completeness": scores_a["completeness"],
            "response_a_clarity": scores_a["clarity"],
            "response_a_conciseness": scores_a["conciseness"],
            "response_a_avg": avg_a,
            # Response B
            "response_b_helpfulness": scores_b["helpfulness"],
            "response_b_accuracy": scores_b["accuracy"],
            "response_b_completeness": scores_b["completeness"],
            "response_b_clarity": scores_b["clarity"],
            "response_b_conciseness": scores_b["conciseness"],
            "response_b_avg": avg_b,
        }

        # Add supervised comparison if enabled
        if supervised_comparison:
            # Determine predicted winner based on average scores
            if avg_a > avg_b:
                predicted_winner = "model_a"
            elif avg_b > avg_a:
                predicted_winner = "model_b"
            else:
                predicted_winner = "tie"

            agreement = 1.0 if predicted_winner == ex["ground_truth_winner"] else 0.0

            print(f"\nPredicted Winner: {predicted_winner}")
            print(f"Agreement: {agreement}")

            log_scores["predicted_winner"] = predicted_winner
            log_scores["agreement"] = agreement

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
                "scores_a": scores_a,
                "scores_b": scores_b,
                "avg_a": avg_a,
                "avg_b": avg_b,
                "predicted_winner": log_scores.get("predicted_winner", None),
            },
            scores=log_scores
        )

    # Log summary
    ev.log_summary()
    print(f"\n{'='*80}")
    print(f"Evaluation complete! View at: {ev.ui_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--weave_project", type=str, default="byyoung3/arena-detailed", help="Weave project name")
    parser.add_argument("--judge_models", type=str, default="gpt-4o", help="Comma-separated judge models (e.g., 'litellm/openai/gpt-4o,litellm/openai/gpt-5')")
    parser.add_argument("--no_supervised_comparison", action="store_true", help="Disable supervised comparison against ground truth")
    args = parser.parse_args()

    run_eval(
        args.num_samples,
        args.seed,
        args.weave_project,
        args.judge_models,
        not args.no_supervised_comparison  # Default to True, disable with flag
    )


