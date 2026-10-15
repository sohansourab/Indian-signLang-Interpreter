import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
SPLITS = ("Training", "Validation", "Testing")
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def load_labels(labels_path):
    with open(labels_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_images(split_dir, labels):
    split_dir = Path(split_dir)
    for label in labels:
        class_dir = split_dir / label
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                yield label, path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash_bytes(path):
    import imagehash
    from PIL import Image

    try:
        with Image.open(path) as image:
            bits = imagehash.phash(image).hash.reshape(-1)
    except Exception:
        return None
    return np.packbits(bits).astype(np.uint8)


def near_duplicate_flags(train_bytes, query_bytes, threshold):
    # True where a query image is within `threshold` Hamming distance of any training image.
    flags = np.zeros(len(query_bytes), dtype=bool)
    for start in range(0, len(query_bytes), 256):
        chunk = query_bytes[start:start + 256]
        distance = POPCOUNT[chunk[:, None, :] ^ train_bytes[None, :, :]].sum(axis=2)
        flags[start:start + 256] = (distance <= threshold).any(axis=1)
    return flags


def scan_split(root, split, labels):
    records = []
    for label, path in iter_images(root / split, labels):
        records.append({"label": label, "path": path, "sha256": sha256(path)})
    return records


def summarize(records):
    by_sha = defaultdict(list)
    per_class_total = defaultdict(int)
    per_class_unique = defaultdict(set)
    for record in records:
        by_sha[record["sha256"]].append(record)
        per_class_total[record["label"]] += 1
        per_class_unique[record["label"]].add(record["sha256"])
    redundant = sum(len(group) - 1 for group in by_sha.values() if len(group) > 1)
    return {
        "total": len(records),
        "unique_sha256": len(by_sha),
        "redundant_within": redundant,
        "per_class": {
            label: {
                "total": per_class_total[label],
                "unique_sha256": len(per_class_unique[label]),
                "redundant_within": per_class_total[label] - len(per_class_unique[label]),
            }
            for label in sorted(per_class_total)
        },
        "_by_sha": by_sha,
    }


def cross_duplicates(records_a, records_b):
    # Images in A whose exact bytes also occur in B, plus per-class counts.
    sha_b = {record["sha256"] for record in records_b}
    matches = [record for record in records_a if record["sha256"] in sha_b]
    per_class = defaultdict(int)
    for record in matches:
        per_class[record["label"]] += 1
    return len(matches), dict(sorted(per_class.items()))


def filename_info(records):
    stems = [record["path"].stem for record in records]
    return {
        "all_numeric_stems": all(stem.isdigit() for stem in stems) if stems else None,
        "example_stems": stems[:5],
        "signer_or_session_identifiers_found": False,
        "note": "Filenames are numeric image indices only; there is no signer or session id, so a group split is not possible from the data.",
    }


def main():
    parser = argparse.ArgumentParser(description="Audit exact and near duplicate images across splits")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--labels", default="labels.json", help="JSON file with class labels")
    parser.add_argument("--output", default="results/duplicate_report.json", help="Where to write the report")
    parser.add_argument("--clean-test", default="splits/clean_test.txt", help="Manifest of clean Testing images")
    parser.add_argument("--threshold", type=int, default=5, help="pHash Hamming distance threshold")
    args = parser.parse_args()

    root = Path(args.root)
    labels = load_labels(root / args.labels)

    records = {split: scan_split(root, split, labels) for split in SPLITS}
    summaries = {split: summarize(records[split]) for split in SPLITS}

    exact_cross = {}
    for a, b in (("Training", "Validation"), ("Training", "Testing"), ("Validation", "Testing")):
        count, per_class = cross_duplicates(records[a], records[b])
        exact_cross[f"{a}_vs_{b}"] = {"count": count, "per_class": per_class}

    # Near-duplicate pHash check: Testing vs Training and Validation vs Training.
    def phash_map(split):
        hashed = []
        unreadable = []
        for record in records[split]:
            value = phash_bytes(record["path"])
            if value is None:
                unreadable.append(record["path"].as_posix())
            else:
                hashed.append(value)
        array = np.array(hashed, dtype=np.uint8) if hashed else np.zeros((0, 8), dtype=np.uint8)
        return array, unreadable

    train_bytes, train_unreadable = phash_map("Training")
    near = {}
    for split in ("Testing", "Validation"):
        query_bytes, query_unreadable = phash_map(split)
        flags = near_duplicate_flags(train_bytes, query_bytes, args.threshold)
        near[f"{split}_vs_Training"] = {
            "near_duplicate_count": int(flags.sum()),
            "compared": int(len(flags)),
            "unreadable_for_phash": len(query_unreadable),
            "unreadable_paths": query_unreadable,
        }

    # Clean-test manifest: Testing images whose exact bytes are absent from Training.
    train_sha = {record["sha256"] for record in records["Training"]}
    clean = [record for record in records["Testing"] if record["sha256"] not in train_sha]
    clean_path = root / args.clean_test
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.write_text(
        "\n".join(str(record["path"].as_posix()) for record in clean) + "\n",
        encoding="utf-8",
    )

    report = {
        "method": {"exact": "sha256", "near": "phash 64-bit", "phash_hamming_threshold": args.threshold},
        "splits": {split: {k: v for k, v in summaries[split].items() if k != "_by_sha"} for split in SPLITS},
        "exact_duplicates_across_splits": exact_cross,
        "near_duplicates_vs_training": near,
        "unreadable_for_phash_training": train_unreadable,
        "clean_test": {
            "manifest": str(args.clean_test),
            "testing_total": len(records["Testing"]),
            "clean_total": len(clean),
            "removed_exact_duplicates": len(records["Testing"]) - len(clean),
        },
        "filenames": filename_info(records["Testing"]),
    }

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {output_path}")
    print(f"Wrote {clean_path} ({len(clean)} clean of {len(records['Testing'])} testing images)")
    print("Exact duplicates across splits:", {k: v["count"] for k, v in exact_cross.items()})
    print("Near duplicates vs Training:", {k: v["near_duplicate_count"] for k, v in near.items()})


if __name__ == "__main__":
    main()
