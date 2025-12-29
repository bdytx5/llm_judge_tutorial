#!/bin/bash

# SimpleQA Evaluation - Single Judge
echo "=========================================="
echo "Running SimpleQA with single judge (GPT-5)"
echo "=========================================="
python ev_simpleQA.py --model litellm/openai/gpt-4o --num_samples 50 --judge_models gpt-5

echo ""
echo "=========================================="
echo "Running SimpleQA with ensemble judges (GPT-5 + Claude Sonnet)"
echo "=========================================="
python ev_simpleQA.py --model litellm/openai/gpt-4o --num_samples 50 --judge_models "litellm/openai/gpt-5,litellm/anthropic/claude-sonnet-4-5-20250929"

# Arena Evaluation - Comparer Pattern
echo ""
echo "=========================================="
echo "Running Arena with single judge (GPT-5)"
echo "=========================================="
python ev_arena.py --num_samples 50 --judge_models gpt-5

echo ""
echo "=========================================="
echo "Running Arena with two judges (GPT-5 + Sonnet)"
echo "=========================================="
python ev_arena.py --num_samples 50 --judge_models "litellm/openai/gpt-5,litellm/anthropic/claude-sonnet-4-5-20250929"


# Arena Detailed Evaluation - Open-ended Pattern
echo ""
echo "=========================================="
echo "Running Arena Detailed with single judge (GPT-5)"
echo "=========================================="
python ev_arena_detailed.py --num_samples 50 --judge_models gpt-5

echo ""
echo "=========================================="
echo "Running Arena Detailed with ensemble judges (GPT-5 + Claude Sonnet)"
echo "=========================================="
python ev_arena_detailed.py --num_samples 50 --judge_models "litellm/openai/gpt-5,litellm/anthropic/claude-sonnet-4-5-20250929"

# JudgeBench Evaluation
echo ""
echo "=========================================="
echo "Running JudgeBench with GPT-5 judge"
echo "=========================================="
python JudgeBench/run_judge_weave.py --pairs JudgeBench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl --judge_name arena_hard --judge_model litellm/openai/gpt-5 --max_samples 50

echo ""
echo "=========================================="
echo "Running JudgeBench with Claude Sonnet judge"
echo "=========================================="
python JudgeBench/run_judge_weave.py --pairs JudgeBench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl --judge_name arena_hard --judge_model litellm/anthropic/claude-sonnet-4-5-20250929 --max_samples 50

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
