"""
Download MathVista splits locally via ModelScope without symlinks.

This script copies every example into a local HuggingFace Dataset saved on disk,
so downstream evaluation (e.g., `mathvista_eval.py`) can load data purely from
local storage. Images are materialized as real files under the output directory
instead of symlinks.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, Features, Image, Value
from modelscope.msdatasets import MsDataset
from tqdm import tqdm


def load_split(split: str, cache_dir: Path) -> List[Dict]:
    ds = MsDataset.load(
        "AI-MO/MathVista",
        split=split,
        cache_dir=str(cache_dir),
    )
    hf_ds = ds.to_hf_dataset()
    records: List[Dict] = []

    for sample in hf_ds:
        qid = str(sample.get("qid", sample.get("id", "unknown")))
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        image = sample.get("image")
        records.append({
            "qid": qid,
            "question": question,
            "answer": answer,
            "image": image,
        })
    return records


def materialize_split(records: List[Dict], split_dir: Path) -> None:
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    processed: List[Dict] = []
    for sample in tqdm(records, desc=f"Saving {split_dir.name}"):
        qid = sample["qid"]
        image_obj = sample["image"]
        image_path = images_dir / f"{qid}.png"

        # Ensure we write a real file (no symlinks) even if source is cached.
        if hasattr(image_obj, "save"):
            image_obj.save(image_path)
        elif isinstance(image_obj, str) and os.path.exists(image_obj):
            shutil.copy2(image_obj, image_path)
        elif isinstance(image_obj, dict) and os.path.exists(image_obj.get("path", "")):
            shutil.copy2(image_obj["path"], image_path)
        else:
            raise ValueError(f"Unrecognized image format for qid={qid}")

        processed.append({
            "qid": qid,
            "question": sample.get("question", ""),
            "answer": sample.get("answer", ""),
            "image": str(image_path),
        })

    features = Features({
        "qid": Value("string"),
        "question": Value("string"),
        "answer": Value("string"),
        "image": Image(),
    })
    dataset = Dataset.from_list(processed, features=features)
    dataset.save_to_disk(str(split_dir))



def save_metadata(output_dir: Path, splits: List[str]) -> None:
    meta_path = output_dir / "mathvista_manifest.json"
    meta = {
        "splits": splits,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MathVista via ModelScope")
    parser.add_argument("--output-dir", required=True, help="Target directory to store the dataset")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["testmini"],
        help="MathVista splits to download (e.g., testmini test)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory for ModelScope downloads",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else output_dir / "_mscache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    available_splits = []
    for split in args.splits:
        records = load_split(split, cache_dir)
        split_dir = output_dir / split
        materialize_split(records, split_dir)
        available_splits.append(split)

    save_metadata(output_dir, available_splits)
    print(f"Saved splits {available_splits} to {output_dir}")


if __name__ == "__main__":
    main()