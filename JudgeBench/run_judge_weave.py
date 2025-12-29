from typing import List, Dict, Any
import argparse
import asyncio
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import weave

import utils.file_operations as file_operations
import utils.judges as judges
import utils.metrics as metrics


def judge_single_pair(
    pair: Dict[str, Any],
    judge_name: str,
    judge_model: str,
    reverse_order: bool = False
) -> Dict[str, Any]:
    """Judge a single pair synchronously."""
    judge = judges.get_judge_from_judge_name_and_model(judge_name, judge_model)

    question = pair["question"]
    response_A = pair["response_A"]
    response_B = pair["response_B"]

    # Make first judgment
    try:
        judgment_1 = asyncio.run(judge.get_judgment(question, response_A, response_B))
    except Exception as e:
        print(f"Failed to judge pair {pair['pair_id']} due to the following error: {e}.")
        judgment_1 = None

    judgments = [judgment_1]

    # Make second judgment if reverse_order is enabled
    if reverse_order:
        try:
            judgment_2 = asyncio.run(judge.get_judgment(question, response_B, response_A))
        except Exception as e:
            print(f"Failed to judge pair {pair['pair_id']} due to the following error: {e}.")
            judgment_2 = None
        judgments.append(judgment_2)

    # Prepare result with all the data we need
    result = {
        "pair": pair,
        "judgments": judgments,
        "inputs": {
            "pair_id": pair["pair_id"],
            "source": pair["source"],
            "question": question,
            "response_A": response_A,
            "response_B": response_B,
            "label": pair.get("label"),
        },
        "output": {
            "judgments": judgments,
        },
        "scores": {}
    }

    # Add decisions to output
    if judgment_1 is not None:
        result["output"]["judgment_1_decision"] = judgment_1.get("decision")

    if reverse_order and judgment_2 is not None:
        result["output"]["judgment_2_decision"] = judgment_2.get("decision")

    # Calculate ACTUAL scores (correctness only)
    if judgment_1 is not None:
        decision_1 = judgment_1.get("decision")
        if "label" in pair and pair["label"] and decision_1:
            result["scores"]["judgment_1_correct"] = 1.0 if decision_1 == pair["label"] else 0.0

    if reverse_order and judgment_2 is not None:
        decision_2 = judgment_2.get("decision")
        if decision_2:
            reversed_decision = decision_2.replace("A", "temp").replace("B", "A").replace("temp", "B")
            if "label" in pair and pair["label"]:
                result["scores"]["judgment_2_correct"] = 1.0 if reversed_decision == pair["label"] else 0.0

    # Add judgment info to pair for local output
    pair["judge_name"] = judge_name
    pair["judgments"] = judgments

    return result


def judge_pairs_with_weave(
    pairs: List[Dict[str, Any]],
    judge_name: str,
    judge_model: str,
    concurrency_limit: int = 1,
    reverse_order: bool = False,
    ev: weave.EvaluationLogger = None
):
    """Judge pairs using ThreadPoolExecutor and log to Weave."""
    results = []

    # Use ThreadPoolExecutor to parallelize judgments
    with ThreadPoolExecutor(max_workers=concurrency_limit) as executor:
        # Submit all tasks
        future_to_pair = {
            executor.submit(judge_single_pair, pair, judge_name, judge_model, reverse_order): pair
            for pair in pairs
        }

        # Collect results as they complete
        for future in tqdm(as_completed(future_to_pair), total=len(pairs), desc="Judging"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                pair = future_to_pair[future]
                print(f"Failed to process pair {pair['pair_id']}: {e}")

    # Now log everything to Weave in one go
    if ev is not None:
        print("Logging results to Weave...")
        for result in tqdm(results, desc="Logging to Weave"):
            try:
                ev.log_example(
                    inputs=result["inputs"],
                    output=result["output"],
                    scores=result["scores"]
                )
            except Exception as e:
                print(f"Failed to log to Weave: {e}")

    # Return pairs with judgments for local saving
    return [result["pair"] for result in results]


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)

    # Initialize Weave if project name is provided
    if args.weave_project:
        weave.init(args.weave_project)
        print(f"Initialized Weave with project: {args.weave_project}")

    pairs = file_operations.read_jsonl(args.pairs)

    # Limit samples if specified
    if args.max_samples:
        pairs = pairs[:args.max_samples]
        print(f"Limited to {len(pairs)} samples for testing")

    dataset_name = os.path.basename(args.pairs).replace(".jsonl", "")
    file_path = f"{dataset_name},judge_name={args.judge_name},judge_model={args.judge_model.replace('/', '_')}.jsonl"
    os.makedirs("./outputs", exist_ok=True)
    file_path = os.path.join("./outputs", file_path)

    # Handle existing results
    if os.path.exists(file_path) and not args.weave_project:
        print(f"File {file_path} already exists. Skipping judging pairs...")
        original_num_pairs = len(pairs)
        existing_pairs = file_operations.read_jsonl(file_path)
        existing_pair_ids = {pair["pair_id"] for pair in existing_pairs}
        pairs = [pair for pair in pairs if pair["pair_id"] not in existing_pair_ids]
        print(f"Skipped {original_num_pairs - len(pairs)} pairs.")

    if pairs:
        print(f"Judging {len(pairs)} pairs ...")

        # Create EvaluationLogger if using Weave
        ev = None
        if args.weave_project:
            ev = weave.EvaluationLogger(
                name=f"{args.judge_name}_{args.judge_model}_{dataset_name}",
                model={"name": f"{args.judge_name}_{args.judge_model}"},
                scorers=["judgment_1_correct", "judgment_2_correct"] if args.double_game else ["judgment_1_correct"],
            )
            print(f"Weave Evaluation URL: {ev.ui_url}")

        # Run the judging
        judged_pairs = judge_pairs_with_weave(
            pairs,
            args.judge_name,
            args.judge_model,
            reverse_order=args.double_game,
            concurrency_limit=args.concurrency_limit,
            ev=ev,
        )

        # Save to local file if not using Weave or if explicitly requested
        if not args.weave_project or args.save_local:
            with open(file_path, 'w') as f:
                for pair in judged_pairs:
                    f.write(weave.util.json_dumps_safe(pair, ensure_ascii=False) + '\n')
            print(f"Saved results to {file_path}")

        # Compute and log summary to Weave
        if ev is not None:
            all_pairs = judged_pairs
            summary = {}

            # Compute metrics per source
            for source in ["mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench", ""]:
                # Filter pairs for this source
                filtered_pairs = [p for p in all_pairs if p["source"].startswith(source)]

                # Skip if no pairs for this source
                if not filtered_pairs:
                    continue

                score = metrics.compute_final_metrics(
                    all_pairs,
                    args.double_game,
                    include_fn=lambda x: x["source"].startswith(source)
                )
                source_name = source if source else 'Overall'
                summary[f"{source_name}_accuracy"] = score
                print(f"{source_name}: {score:.2f}%.")

            # Log summary to Weave
            ev.log_summary(summary=summary)
            print(f"Logged summary to Weave: {ev.ui_url}")
    else:
        print("No pairs to judge.")

    # Compute final metrics from file if available
    if os.path.exists(file_path) and not args.weave_project:
        print("Computing final metrics from saved file...")
        pairs = file_operations.read_jsonl(file_path)
        for source in ["mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench", ""]:
            # Filter pairs for this source
            filtered_pairs = [p for p in pairs if p["source"].startswith(source)]

            # Skip if no pairs for this source
            if not filtered_pairs:
                continue

            score = metrics.compute_final_metrics(
                pairs,
                args.double_game,
                include_fn=lambda x: x["source"].startswith(source)
            )
            print(f"{source if source else 'Overall'}: {score:.2f}%.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--judge_name', type=str, required=True, help='Name of judge')
    parser.add_argument('--judge_model', type=str, required=True, help='Model to be used by judge')
    parser.add_argument('--double_game', action="store_true", help='Run with reversed order to check position bias (runs twice per pair)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--concurrency_limit', type=int, default=1, help='Number of concurrent judgments')
    parser.add_argument('--pairs', type=str, required=True, help='Path to jsonl containing pairs')
    parser.add_argument('--weave_project', type=str, help='Weave project name (e.g., "your-entity/judge-bench")')
    parser.add_argument('--save_local', action="store_true", help='Save to local file even when using Weave')
    parser.add_argument('--max_samples', type=int, help='Limit evaluation to first N samples (for testing)')
    args = parser.parse_args()
    main(args)
