#!/usr/bin/env python3
"""
Run :class:`chain.reasoning_chain.ReasoningChain` over ``train.csv`` and append
:class:`chain.data_types.DataPoint` records as JSON lines.

Local usage (repo root)::

    PYTHONPATH=src python scripts/run_reasoning_chain_jsonl.py --backend hf --hf-model <hf_id> --limit 2

Kaggle: attach data, enable GPU as needed, set ``--output /kaggle/working/results.jsonl`` (default when that
directory exists). Re-run to resume: existing output IDs are skipped unless ``--no-resume``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from chain import ReasoningChain, convert_to_entry  # noqa: E402
from registry.datasets import get_train_csv_path  # noqa: E402
from registry.models import get_model, get_transformers_chat_model  # noqa: E402


def _default_output_path() -> Path:
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working / "reasoning_chain_results.jsonl"
    return Path("reasoning_chain_results.jsonl")


def _load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(str(json.loads(line)["id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ReasoningChain → JSONL")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="Path to train.csv (default: Kaggle input or kagglehub download)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"JSONL output path (default: {_default_output_path()})",
    )
    parser.add_argument("--backend", choices=("ollama", "hf"), required=True)
    parser.add_argument("--ollama-model-id", default="nemotron3-4b")
    parser.add_argument("--hf-model", default=None, help="HF pretrained id or path (required for --backend hf)")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip IDs already in output")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    train_path = args.train_csv if args.train_csv is not None else get_train_csv_path()
    out_path = args.output if args.output is not None else _default_output_path()

    if args.backend == "hf" and not args.hf_model:
        parser.error("--hf-model is required when --backend hf")

    if args.backend == "ollama":
        model = get_model(args.ollama_model_id)
    else:
        model = get_transformers_chat_model(
            args.hf_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            trust_remote_code=args.trust_remote_code,
        )

    df = pl.read_csv(train_path)
    if args.limit is not None:
        df = df.head(args.limit)

    done = set() if args.no_resume else _load_done_ids(out_path)
    chain = ReasoningChain(model, verbose=args.verbose)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        for row in df.iter_rows(named=True):
            eid = str(row["id"]).strip()
            if eid in done:
                continue
            entry = convert_to_entry(pl.DataFrame([row]))
            dp = chain.run(entry)
            fh.write(json.dumps(dp.model_dump(), ensure_ascii=False) + "\n")
            fh.flush()
            done.add(eid)


if __name__ == "__main__":
    main()
