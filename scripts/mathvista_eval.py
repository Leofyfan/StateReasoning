"""
MathVista evaluation with VERL and confidence tracing.

This script runs Qwen3-VL-4B-Thinking on the MathVista dataset and logs
inference trajectories to Weights & Biases. It captures both logit-based
confidence and a simple relative confidence heuristic to study how model
"state" evolves during reasoning.
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
)


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


def build_prompt(question: str) -> str:
    return (
        "You are a careful math and vision tutor. Read the question and image, "
        "think step by step, and output a concise final answer."
        "\nQuestion: "
        f"{question}\nAnswer:"
    )


def extract_confidence(
    generation_outputs: torch.LongTensor,
    scores: List[torch.Tensor],
    input_length: int,
) -> ConfidenceSummary:
    sequence = generation_outputs[:, input_length:]
    token_confidences: List[float] = []
    margin_confidences: List[float] = []

    for step, score in enumerate(scores):
        token_id = sequence[0, step]
        probs = torch.softmax(score[0], dim=-1)
        prob = probs[token_id].item()
        token_confidences.append(prob)
        top2 = torch.topk(probs, 2).values
        margin_confidences.append((top2[0] - top2[1]).item())

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
    model,
    processor,
    device: torch.device,
    sample: Dict,
    max_new_tokens: int,
) -> SampleResult:
    image: Image.Image = sample["image"]
    prompt = build_prompt(sample["question"])

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(device)

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)

    text = processor.batch_decode(outputs.sequences[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    confidence = extract_confidence(outputs.sequences, outputs.scores, inputs.input_ids.shape[1])

    return SampleResult(
        question_id=str(sample.get("qid", sample.get("id", "unknown"))),
        question=sample["question"],
        answer=sample.get("answer", ""),
        prediction=text.strip(),
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
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    dataset = load_dataset("AI-MO/MathVista", data_dir=data_dir, split=split)
    if limit:
        dataset = dataset.select(range(limit))

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
                model=model,
                processor=processor,
                device=device,
                sample=sample,
                max_new_tokens=max_new_tokens,
            )
            log_to_wandb(table, result)

    wandb.log({"mathvista_eval": table})
    wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MathVista evaluation with VERL-style logging")
    parser.add_argument("--model-id", default="/home/shenyl/hf/model/Qwen/Qwen3-VL-4B-Thinking", help="Model identifier")
    parser.add_argument("--split", default="testmini", help="Dataset split to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--project", default="StateReasoning", help="Weights & Biases project name")
    parser.add_argument("--run-name", default="qwen3-vl-4b-thinking", help="Weights & Biases run name")
    parser.add_argument("--data-dir", default="/home/yuanfan/projects/StateReasoning/data", help="Optional local path to MathVista data")
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
    )


if __name__ == "__main__":
    main()
