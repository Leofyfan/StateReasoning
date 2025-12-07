"""
MathVista evaluation with VERL and confidence tracing.

This script runs Qwen3-VL-4B-Thinking on the MathVista dataset with vLLM
(0.11.0) and logs inference trajectories to Weights & Biases. It captures
both logit-based confidence and a simple relative confidence heuristic to
study how model "state" evolves during reasoning.
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import wandb
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


PHASES = [
    "read",
    "explore",
    "insight",
    "solve",
    "verify",
]


@dataclass
class ConfidenceSummary:
    absolute: float
    relative: float
    phase_curve: List[Tuple[str, float]]


@dataclass
class SampleResult:
    question_id: str
    question: str
    answer: str
    prediction: str
    confidence: ConfidenceSummary
    trajectory: List[Dict[str, float]]


def build_messages(question: str, image: Image.Image) -> List[Dict]:
    return [
        {
            "role": "system",
            "content": "You are a careful math and vision tutor. Read the question and image, "
            "think step by step, and output a concise final answer.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{question}"},
            ],
        },
    ]


def extract_confidence(
    token_ids: List[int],
    logprob_trace: List[Dict],
) -> ConfidenceSummary:
    """Derive absolute and relative confidence from vLLM logprobs."""

    token_confidences: List[float] = []
    margin_confidences: List[float] = []

    for step, token_id in enumerate(token_ids):
        step_logprobs = logprob_trace[step] if step < len(logprob_trace) else {}
        probs: List[float] = []
        target_prob: Optional[float] = None

        for entry in step_logprobs.values():
            if isinstance(entry, (float, int)):
                logprob = float(entry)
                entry_token_id = None
            else:
                logprob = getattr(entry, "logprob", None)
                entry_token_id = getattr(entry, "token_id", None)

            if logprob is None:
                continue

            prob = float(torch.exp(torch.tensor(logprob)).item())
            probs.append(prob)
            if entry_token_id == token_id:
                target_prob = prob

        if target_prob is None and probs:
            target_prob = probs[0]

        token_confidences.append(target_prob or 0.0)
        if len(probs) >= 2:
            top_two = sorted(probs, reverse=True)[:2]
            margin_confidences.append(top_two[0] - top_two[1])
        elif probs:
            margin_confidences.append(probs[0])
        else:
            margin_confidences.append(0.0)

    avg_prob = sum(token_confidences) / max(len(token_confidences), 1)
    absolute = float(avg_prob)
    relative = float(sum(margin_confidences) / max(len(margin_confidences), 1))
    phase_curve = interpolate_phases(token_confidences)
    return ConfidenceSummary(absolute=absolute, relative=relative, phase_curve=phase_curve)


def interpolate_phases(token_confidences: List[float]) -> List[Tuple[str, float]]:
    if not token_confidences:
        return [(phase, 0.0) for phase in PHASES]

    min_conf = min(token_confidences)
    max_conf = max(token_confidences)
    span = max(max_conf - min_conf, 1e-6)
    normalized = [(c - min_conf) / span for c in token_confidences]

    checkpoints = []
    for idx, phase in enumerate(PHASES):
        pos = int((idx / max(len(PHASES) - 1, 1)) * (len(normalized) - 1))
        checkpoints.append((phase, float(normalized[pos])))
    return checkpoints


def run_sample(
    llm: LLM,
    processor,
    sample: Dict,
    sampling_params: SamplingParams,
) -> SampleResult:
    image: Image.Image = sample["image"]
    messages = build_messages(sample["question"], image)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    outputs = llm.generate(
        {"prompt": prompt, "multi_modal_data": {"image": image}},
        sampling_params=sampling_params,
    )

    result = outputs[0].outputs[0]
    logprob_trace = result.logprobs or []
    confidence = extract_confidence(result.token_ids, logprob_trace)

    return SampleResult(
        question_id=str(sample.get("qid", sample.get("id", "unknown"))),
        question=sample["question"],
        answer=sample.get("answer", ""),
        prediction=result.text.strip(),
        confidence=confidence,
        trajectory=[{"phase": phase, "confidence": value} for phase, value in confidence.phase_curve],
    )


def log_to_wandb(table: wandb.Table, result: SampleResult) -> None:
    table.add_data(
        result.question_id,
        result.question,
        result.answer,
        result.prediction,
        result.confidence.absolute,
        result.confidence.relative,
        result.trajectory,
    )


def evaluate(
    model_id: str,
    split: str,
    limit: Optional[int],
    batch_size: int,
    max_new_tokens: int,
    project: str,
    run_name: str,
    data_dir: Optional[str] = None,
    dtype: str = "bfloat16",
) -> None:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        dtype=dtype,
    )

    dataset = load_dataset("AI-MO/MathVista", data_dir=data_dir, split=split)
    if limit:
        dataset = dataset.select(range(limit))

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        logprobs=10,
    )

    wandb.init(project=project, name=run_name, config={
        "model_id": model_id,
        "split": split,
        "limit": limit,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
    })

    table = wandb.Table(
        columns=[
            "question_id",
            "question",
            "ground_truth",
            "prediction",
            "absolute_confidence",
            "relative_confidence",
            "phase_trajectory",
        ]
    )

    for start in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[start : start + batch_size]
        for sample in batch:
            result = run_sample(
                llm=llm,
                processor=processor,
                sample=sample,
                sampling_params=sampling_params,
            )
            log_to_wandb(table, result)

    wandb.log({"mathvista_eval": table})
    wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MathVista evaluation with VERL-style logging")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Thinking", help="Model identifier")
    parser.add_argument("--split", default="testmini", help="Dataset split to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--project", default="mathvista-verl", help="Weights & Biases project name")
    parser.add_argument("--run-name", default="qwen3-vl-4b-thinking", help="Weights & Biases run name")
    parser.add_argument("--data-dir", default=None, help="Optional local path to MathVista data")
    parser.add_argument("--dtype", default="bfloat16", help="Computation dtype for vLLM (e.g., float16, bfloat16)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        model_id=args.model_id,
        split=args.split,
        limit=args.limit,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        project=args.project,
        run_name=args.run_name,
        data_dir=args.data_dir,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
