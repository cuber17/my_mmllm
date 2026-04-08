import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np


VIEW_KEYS = ("td_path", "tr_path", "ta_path")
SPLIT_FILES = ("train.json", "test.json")
AUG0_TOKEN = "_aug0"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_rel_path(dataset_root: Path, rel_path: str) -> Path:
    clean = rel_path[2:] if rel_path.startswith("./") else rel_path
    return dataset_root / clean


def array_is_constant(arr: np.ndarray, eps: float = 1e-8) -> bool:
    return float(arr.std()) < eps


def split_aug0_filename(filename: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # Example: 005783_td_aug0.npy -> ("005783", "td", ".npy")
    name = Path(filename).name
    if not name.endswith(".npy") or AUG0_TOKEN not in name:
        return None, None, None
    stem = name[:-4]  # remove .npy
    if not stem.endswith(AUG0_TOKEN):
        return None, None, None
    core = stem[: -len(AUG0_TOKEN)]
    if "_" not in core:
        return None, None, None
    sample_id, view = core.rsplit("_", 1)
    if view not in {"td", "tr", "ta"}:
        return None, None, None
    return sample_id, view, ".npy"


def build_candidate_paths(dataset_root: Path, target_filename: str) -> Iterable[Path]:
    sample_id, view, ext = split_aug0_filename(target_filename)
    if sample_id is None:
        return []

    folders = ("imgs", "imgs_test")
    candidates = []

    # 1) Prefer clean/original signal without aug suffix.
    base_name = f"{sample_id}_{view}{ext}"
    for folder in folders:
        candidates.append(dataset_root / folder / base_name)

    # 2) Fallback to aug1/aug2 from either folder if needed.
    for aug in ("aug1", "aug2"):
        aug_name = f"{sample_id}_{view}_{aug}{ext}"
        for folder in folders:
            candidates.append(dataset_root / folder / aug_name)

    return candidates


def find_reference_array(dataset_root: Path, target_file: Path, eps: float) -> Optional[np.ndarray]:
    candidates = build_candidate_paths(dataset_root, target_file.name)
    for cand in candidates:
        if not cand.exists() or cand.resolve() == target_file.resolve():
            continue
        try:
            arr = np.load(cand)
        except Exception:
            continue
        if arr.size == 0 or array_is_constant(arr, eps=eps):
            continue
        return arr
    return None


def add_salt_pepper_noise(
    arr: np.ndarray,
    noise_ratio: float,
    salt_vs_pepper: float,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.array(arr, copy=True)
    if out.size == 0:
        return out

    h, w = out.shape[:2]
    total = h * w
    n_noisy = max(1, int(total * noise_ratio))

    flat = out.reshape(total, *out.shape[2:])
    indices = rng.choice(total, size=n_noisy, replace=False)

    n_salt = int(n_noisy * salt_vs_pepper)
    salt_indices = indices[:n_salt]
    pepper_indices = indices[n_salt:]

    min_val = float(np.min(out))
    max_val = float(np.max(out))
    if abs(max_val - min_val) < 1e-12:
        max_val = min_val + 1.0

    if salt_indices.size > 0:
        flat[salt_indices] = max_val
    if pepper_indices.size > 0:
        flat[pepper_indices] = min_val

    repaired = flat.reshape(out.shape)

    # Guard against accidental constant output.
    if float(repaired.std()) < 1e-8:
        repaired = repaired.astype(np.float32, copy=False)
        repaired.flat[0] = min_val
        repaired.flat[-1] = max_val

    return repaired.astype(arr.dtype, copy=False)


def collect_aug0_paths(dataset_root: Path) -> Set[Path]:
    aug0_paths: Set[Path] = set()
    for split_name in SPLIT_FILES:
        split_path = dataset_root / split_name
        if not split_path.exists():
            continue

        items = load_json(split_path)
        for item in items:
            for key in VIEW_KEYS:
                rel = item.get(key)
                if not isinstance(rel, str):
                    continue
                if AUG0_TOKEN not in rel or not rel.endswith(".npy"):
                    continue
                aug0_paths.add(resolve_rel_path(dataset_root, rel))
    return aug0_paths


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Repair corrupted *_aug0.npy radar files by reconstructing from a valid reference "
            "(base/aug1/aug2) and injecting salt-pepper noise."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/root/jyz/my_mmLLM/processed_dataset",
        help="Dataset root containing train.json/test.json and imgs/imgs_test folders.",
    )
    parser.add_argument(
        "--noise-ratio",
        type=float,
        default=0.02,
        help="Fraction of pixels to perturb with salt-pepper noise (default: 0.02).",
    )
    parser.add_argument(
        "--salt-ratio",
        type=float,
        default=0.5,
        help="Portion of noisy pixels set to max value (default: 0.5).",
    )
    parser.add_argument(
        "--constant-eps",
        type=float,
        default=1e-8,
        help="Std threshold to judge whether an array is constant.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic repair.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be repaired without writing files.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if not (0 < args.noise_ratio <= 1.0):
        raise ValueError("--noise-ratio must be in (0, 1].")
    if not (0.0 <= args.salt_ratio <= 1.0):
        raise ValueError("--salt-ratio must be in [0, 1].")

    rng = np.random.default_rng(args.seed)

    aug0_paths = sorted(collect_aug0_paths(dataset_root))
    print(f"Found {len(aug0_paths)} unique aug0 files from train/test indices.")

    repaired = 0
    skipped_nonconstant = 0
    missing_target = 0
    missing_reference = 0

    for target in aug0_paths:
        if not target.exists():
            missing_target += 1
            continue

        try:
            current = np.load(target)
        except Exception as e:
            print(f"[WARN] Failed to load target: {target} ({e})")
            continue

        if not array_is_constant(current, eps=args.constant_eps):
            skipped_nonconstant += 1
            continue

        ref = find_reference_array(dataset_root, target, eps=args.constant_eps)
        if ref is None:
            missing_reference += 1
            print(f"[WARN] No valid reference found for: {target}")
            continue

        repaired_arr = add_salt_pepper_noise(
            ref,
            noise_ratio=args.noise_ratio,
            salt_vs_pepper=args.salt_ratio,
            rng=rng,
        )

        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, repaired_arr)

        repaired += 1

    print("\n=== Repair Summary ===")
    print(f"Repaired files          : {repaired}")
    print(f"Skipped (already normal): {skipped_nonconstant}")
    print(f"Missing target files    : {missing_target}")
    print(f"Missing references      : {missing_reference}")
    print(f"Dry run                 : {args.dry_run}")


if __name__ == "__main__":
    main()
