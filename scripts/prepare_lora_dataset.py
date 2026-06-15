from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split persona SFT JSONL into train/val/test files.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="persona")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--val-ratio", default=0.05, type=float)
    parser.add_argument("--test-ratio", default=0.05, type=float)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    random.Random(args.seed).shuffle(rows)
    n_total = len(rows)
    n_test = int(n_total * args.test_ratio)
    n_val = int(n_total * args.val_ratio)

    test = rows[:n_test]
    val = rows[n_test : n_test + n_val]
    train = rows[n_test + n_val :]

    train_path = args.output_dir / f"{args.prefix}.train.jsonl"
    val_path = args.output_dir / f"{args.prefix}.val.jsonl"
    test_path = args.output_dir / f"{args.prefix}.test.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    write_jsonl(test_path, test)

    report = {
        "input": str(args.input),
        "total": n_total,
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "seed": args.seed,
        "format": "chat_messages_jsonl",
        "files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    }
    report_path = args.output_dir / f"{args.prefix}.split-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
