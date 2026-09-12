"""Export logged interactions as JSONL for evaluation or fine-tuning.
Run:  python -m scripts.export_dataset [out.jsonl]
"""
import sys

from core.memory import MemoryStore


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "dataset.jsonl"
    n = MemoryStore().export_jsonl(out)
    print(f"Exported {n} interactions to {out}")


if __name__ == "__main__":
    main()
