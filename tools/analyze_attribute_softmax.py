import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attributes_perception.dataset import MMWaveAttributeDataset
from src.attributes_perception.model import MultiHeadAttributeClassifier


TASKS = ["action_category", "posture", "intensity", "active_part", "trajectory"]


def resolve_dataset_root(project_root: Path) -> Path:
    candidates = [
        project_root / "processed_dataset_rebalanced",
        project_root / "processed_dataset",
    ]

    for root in candidates:
        train_json = root / "train.json"
        has_images = (root / "imgs").exists() or (root / "imgs_test").exists()
        if train_json.exists() and has_images:
            return root

    raise FileNotFoundError(
        "No valid dataset root found. Need train.json and imgs/imgs_test under processed_dataset_rebalanced or processed_dataset."
    )


def load_label_maps(exp_dir: Path) -> Dict[str, Dict[str, int]]:
    label_map_path = exp_dir / "label_maps.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map not found: {label_map_path}")
    with open(label_map_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(json_path: Path, dataset_root: Path, label_maps: Dict[str, Dict[str, int]]):
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return MMWaveAttributeDataset(
        json_file=str(json_path),
        root_dir=str(dataset_root),
        transform=transform,
        label_maps=label_maps,
    )


@torch.no_grad()
def collect_stats(
    model,
    dataloader,
    device: torch.device,
    task_names: List[str],
    temperature: float,
) -> Dict[str, dict]:
    stats = {
        task: {
            "confidences": [],
            "correct_confidences": [],
            "wrong_confidences": [],
            "correct_flags": [],
            "preds": [],
            "targets": [],
        }
        for task in task_names
    }

    model.eval()
    for imgs, labels_dict in tqdm(dataloader, desc="Collecting", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in labels_dict.items()}
        outputs = model(imgs)

        for task in task_names:
            logits = outputs[task]
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=1)
            confs, preds = probs.max(dim=1)
            gold = targets[task]
            correct = preds.eq(gold)

            stats[task]["confidences"].extend(confs.detach().cpu().tolist())
            stats[task]["correct_flags"].extend(correct.detach().cpu().tolist())
            stats[task]["preds"].extend(preds.detach().cpu().tolist())
            stats[task]["targets"].extend(gold.detach().cpu().tolist())

            correct_confs = confs[correct]
            wrong_confs = confs[~correct]
            if correct_confs.numel() > 0:
                stats[task]["correct_confidences"].extend(correct_confs.detach().cpu().tolist())
            if wrong_confs.numel() > 0:
                stats[task]["wrong_confidences"].extend(wrong_confs.detach().cpu().tolist())

    return stats


def summarize_array(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "p05": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p05": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def threshold_sweep(confidences: List[float], correct_flags: List[bool], thresholds: List[float]) -> List[Dict[str, float]]:
    arr = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct_flags, dtype=bool)
    rows = []
    total = int(arr.size)
    for t in thresholds:
        accept = arr >= t
        accepted = int(accept.sum())
        if accepted == 0:
            accepted_acc = 0.0
        else:
            accepted_acc = float(corr[accept].mean())
        coverage = float(accepted / total) if total else 0.0
        risk = float(1.0 - accepted_acc) if accepted else 0.0
        rows.append(
            {
                "threshold": float(t),
                "coverage": coverage,
                "accepted": accepted,
                "accepted_accuracy": accepted_acc,
                "reject_rate": float(1.0 - coverage),
                "accept_error_rate": risk,
            }
        )
    return rows


def pick_threshold(rows: List[Dict[str, float]], target_accuracy: float) -> Dict[str, float]:
    feasible = [r for r in rows if r["accepted"] > 0 and r["accepted_accuracy"] >= target_accuracy]
    if feasible:
        # 在满足目标精度的前提下，优先保留更多 coverage；coverage 相同则选择更高阈值更稳妥
        feasible = sorted(feasible, key=lambda r: (r["coverage"], r["threshold"]), reverse=True)
        return feasible[0]

    # 如果没有阈值能达到目标精度，就选择 accepted_accuracy / coverage 的折中点
    if rows:
        return sorted(rows, key=lambda r: (r["accepted_accuracy"], r["coverage"], r["threshold"]), reverse=True)[0]
    return {"threshold": 0.0, "coverage": 0.0, "accepted": 0, "accepted_accuracy": 0.0, "reject_rate": 1.0, "accept_error_rate": 0.0}


def print_task_report(task: str, task_label_names: List[str], stats: Dict[str, dict], thresholds: List[float], target_accuracy: float):
    confidences = stats[task]["confidences"]
    correct_flags = stats[task]["correct_flags"]
    acc = float(np.mean(correct_flags)) if correct_flags else 0.0

    print(f"\n=== {task} ===")
    print(f"Samples            : {len(confidences)}")
    print(f"Top-1 Accuracy      : {acc * 100:.2f}%")
    print(f"Classes             : {len(task_label_names)}")

    overall = summarize_array(confidences)
    correct = summarize_array(stats[task]["correct_confidences"])
    wrong = summarize_array(stats[task]["wrong_confidences"])

    print("Softmax top-1 confidence distribution:")
    print(
        "  overall -> "
        f"mean={overall['mean']:.4f}, median={overall['median']:.4f}, p10={overall['p10']:.4f}, p25={overall['p25']:.4f}, "
        f"p75={overall['p75']:.4f}, p90={overall['p90']:.4f}, p95={overall['p95']:.4f}, max={overall['max']:.4f}"
    )
    print(
        "  correct -> "
        f"mean={correct['mean']:.4f}, median={correct['median']:.4f}, p10={correct['p10']:.4f}, p25={correct['p25']:.4f}, "
        f"p75={correct['p75']:.4f}, p90={correct['p90']:.4f}, p95={correct['p95']:.4f}, max={correct['max']:.4f}"
    )
    print(
        "  wrong   -> "
        f"mean={wrong['mean']:.4f}, median={wrong['median']:.4f}, p10={wrong['p10']:.4f}, p25={wrong['p25']:.4f}, "
        f"p75={wrong['p75']:.4f}, p90={wrong['p90']:.4f}, p95={wrong['p95']:.4f}, max={wrong['max']:.4f}"
    )

    sweep = threshold_sweep(confidences, correct_flags, thresholds)
    best = pick_threshold(sweep, target_accuracy=target_accuracy)

    print(f"Recommended threshold for target accepted accuracy >= {target_accuracy * 100:.1f}%:")
    print(
        f"  threshold={best['threshold']:.2f}, coverage={best['coverage'] * 100:.2f}%, "
        f"accepted_accuracy={best['accepted_accuracy'] * 100:.2f}%"
    )
    print("Threshold sweep:")
    for row in sweep:
        print(
            f"  t={row['threshold']:.2f} | coverage={row['coverage'] * 100:.2f}% | "
            f"accepted_acc={row['accepted_accuracy'] * 100:.2f}% | accepted={row['accepted']}"
        )

    return {
        "samples": len(confidences),
        "top1_accuracy": acc,
        "confidence_overall": overall,
        "confidence_correct": correct,
        "confidence_wrong": wrong,
        "threshold_sweep": sweep,
        "recommended": best,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze softmax confidence distribution for an attribute prediction checkpoint and recommend thresholds."
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default="/root/jyz/my_mmLLM",
        help="Project root path.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="attributes_20260406_090119",
        help="Attribute experiment folder name under logs/.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to analyze.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu.",
    )
    parser.add_argument(
        "--target-accepted-accuracy",
        type=float,
        default=0.95,
        help="Target accepted accuracy for threshold selection.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
        help="Comma-separated threshold candidates.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to save the analysis result as JSON.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=3.0,
        help="Softmax temperature T for analysis. Probabilities are computed as softmax(logits / T).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(args.project_root)
    exp_dir = project_root / "logs" / args.experiment_id
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    dataset_root = resolve_dataset_root(project_root)
    json_path = dataset_root / f"{args.split}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Split file not found: {json_path}")

    label_maps = load_label_maps(exp_dir)
    output_dims = {k: len(v) for k, v in label_maps.items()}

    eval_device = args.device
    if eval_device == "cuda" and not torch.cuda.is_available():
        eval_device = "cpu"
        print("CUDA is not available, fallback to CPU.")

    device = torch.device(eval_device)
    if args.temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {args.temperature}")

    print("=== Config ===")
    print(f"Project root   : {project_root}")
    print(f"Experiment id  : {args.experiment_id}")
    print(f"Checkpoint     : {exp_dir / 'best.pth'}")
    print(f"Dataset root    : {dataset_root}")
    print(f"Split          : {args.split}")
    print(f"Device         : {device}")
    print(f"Temperature    : {args.temperature}")
    print(f"Output dims    : {output_dims}")

    dataset = build_dataset(json_path, dataset_root, label_maps)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=(device.type == "cuda"),
    )

    model = MultiHeadAttributeClassifier(output_dims).to(device)
    ckpt_path = exp_dir / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    task_names = [k for k in TASKS if k in output_dims]
    stats = collect_stats(
        model,
        dataloader,
        device,
        task_names,
        temperature=float(args.temperature),
    )

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    results = {
        "experiment_id": args.experiment_id,
        "split": args.split,
        "dataset_root": str(dataset_root),
        "checkpoint": str(ckpt_path),
        "temperature": float(args.temperature),
        "target_accepted_accuracy": args.target_accepted_accuracy,
        "tasks": {},
    }

    for task in task_names:
        task_names_for_labels = list(label_maps.get(task, {}).keys())
        results["tasks"][task] = print_task_report(
            task,
            task_names_for_labels,
            stats,
            thresholds,
            target_accuracy=args.target_accepted_accuracy,
        )

    if args.output_json:
        out_path = Path(args.output_json)
    else:
        out_path = exp_dir / f"softmax_analysis_{args.split}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved analysis to: {out_path}")


if __name__ == "__main__":
    main()
