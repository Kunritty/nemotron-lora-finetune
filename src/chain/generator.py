"""
Run a reasoning chain over an entire table, with optional CSV checkpoints for long runs.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import polars as pl
from langchain_ollama import ChatOllama

from .data_types import DataPoint
from .pipeline import MessageStep
from .reasoning_chain import ReasoningChain
from .utils import entry_from_mapping

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = frozenset({"id", "prompt", "answer"})


def validate_dataset_columns(df: pl.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")


def _id_set_from_checkpoint(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return set(pl.read_csv(path, columns=["id"])["id"].cast(pl.Utf8).to_list())


def _append_csv_rows(path: Path, rows: list[dict[str, Any]], *, write_header: bool) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    mode = "w" if write_header else "a"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def datapoints_to_dataframe(points: list[DataPoint]) -> pl.DataFrame:
    if not points:
        return pl.DataFrame(
            schema={
                "id": pl.Utf8,
                "category": pl.Utf8,
                "prompt": pl.Utf8,
                "answer": pl.Utf8,
                "reasoning": pl.Utf8,
                "final_answer": pl.Utf8,
                "confidence": pl.Float64,
                "is_correct": pl.Boolean,
            }
        )
    return pl.DataFrame([p.model_dump() for p in points])


def export_dataframe_csv(df: pl.DataFrame, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


class DatasetGenerator:
    """Applies :class:`ReasoningChain` row-wise with batched disk checkpoints."""

    def __init__(
        self,
        model: ChatOllama,
        *,
        steps: list[MessageStep] | tuple[MessageStep, ...] | None = None,
        system_prompt: str | None = None,
        verify_system_prompt: str | None = None,
        improve_system_prompt: str | None = None,
    ):
        if steps is not None:
            self._chain = ReasoningChain(model, steps=steps)
        else:
            self._chain = ReasoningChain(
                model,
                system_prompt=system_prompt,
                verify_system_prompt=verify_system_prompt,
                improve_system_prompt=improve_system_prompt,
            )

    def run(
        self,
        dataset: pl.DataFrame,
        *,
        limit: int | None = None,
        offset: int = 0,
        checkpoint_path: Path | str | None = None,
        checkpoint_every: int = 100,
        resume: bool = True,
        progress_every: int = 50,
    ) -> pl.DataFrame:
        """
        Process ``dataset`` rows sequentially (friendly to a single local Ollama worker).

        Parameters
        ----------
        dataset:
            Must include columns ``id``, ``prompt``, ``answer``. Optional ``category``.
        limit:
            Maximum rows to process after ``offset`` (None = all remaining).
        offset:
            Skip this many rows from the start before applying ``limit``.
        checkpoint_path:
            If set, append CSV batches every ``checkpoint_every`` rows. Use with
            ``resume=True`` to skip ids already present in the file.
        checkpoint_every:
            Rows between append flushes (bounds memory and preserves progress).
        resume:
            When True and ``checkpoint_path`` exists, skip ids already in that file.
        progress_every:
            Emit a log line every N processed *new* rows.
        """
        validate_dataset_columns(dataset)
        if offset < 0:
            raise ValueError("offset must be >= 0")
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0 when provided")
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1")

        df = dataset.slice(offset, limit if limit is not None else len(dataset) - offset)
        ck_path = Path(checkpoint_path) if checkpoint_path else None

        if ck_path and not resume and ck_path.is_file():
            ck_path.unlink()

        done_ids: set[str] = set()
        if ck_path and resume:
            done_ids = _id_set_from_checkpoint(ck_path)

        if done_ids:
            pending = df.filter(~pl.col("id").cast(pl.Utf8).is_in(list(done_ids)))
            logger.info("Resume: skipping %s rows already in checkpoint", len(done_ids))
        else:
            pending = df

        if pending.is_empty():
            if ck_path and ck_path.is_file():
                return pl.read_csv(ck_path)
            return pl.DataFrame(
                schema={
                    "id": pl.Utf8,
                    "category": pl.Utf8,
                    "prompt": pl.Utf8,
                    "answer": pl.Utf8,
                    "reasoning": pl.Utf8,
                    "final_answer": pl.Utf8,
                    "confidence": pl.Float64,
                    "is_correct": pl.Boolean,
                }
            )

        write_header = not (ck_path and ck_path.is_file())
        buffer: list[dict[str, Any]] = []
        collected: list[dict[str, Any]] | None = None if ck_path else []
        processed = 0
        total_pending = pending.height

        for row in pending.iter_rows(named=True):
            entry = entry_from_mapping(row)
            dp = self._chain.run(entry)
            rec = dp.model_dump()
            if collected is not None:
                collected.append(rec)
            if ck_path:
                buffer.append(rec)
                if len(buffer) >= checkpoint_every:
                    _append_csv_rows(ck_path, buffer, write_header=write_header)
                    write_header = False
                    buffer.clear()

            processed += 1
            if progress_every > 0 and processed % progress_every == 0:
                logger.info("Processed %s / %s rows", processed, total_pending)

        if ck_path and buffer:
            _append_csv_rows(ck_path, buffer, write_header=write_header)

        if ck_path and ck_path.is_file():
            return pl.read_csv(ck_path)

        assert collected is not None
        return pl.DataFrame(collected)
