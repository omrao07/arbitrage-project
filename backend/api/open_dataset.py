# backend/data/open_dataset.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


class OpenDataset:
    """
    Minimal appendable Parquet dataset for small/medium public sets.

    - Ensures a concrete filesystem path (no Optional[str] issues)
    - Creates parent directories on first write
    - Appends by read -> concat -> write (simple & safe)
      (Good for small/medium files; switch to a writer if you outgrow this)
    """

    def __init__(
        self,
        parquet_path: str | os.PathLike[str],
        expected_schema: Optional[pa.Schema] = None,
    ) -> None:
        if parquet_path is None:  # <-- fixes your None path error
            raise ValueError("parquet_path cannot be None")

        self._parquet_path: Path = Path(parquet_path)
        self._expected_schema: Optional[pa.Schema] = expected_schema

    # ---------- internal helpers ----------

    def _ensure_parent_dir(self) -> None:
        self._parquet_path.parent.mkdir(parents=True, exist_ok=True)

    def _coerce_rows_to_table(self, rows: Sequence[Mapping[str, Any]] | Mapping[str, Iterable[Any]]) -> pa.Table:
        """
        Accepts:
          - list[dict]  -> pa.Table.from_pylist
          - dict of columns -> pa.Table.from_pydict
        """
        if isinstance(rows, Mapping):
            return pa.Table.from_pydict(rows)  # columns dict
        if not isinstance(rows, Sequence):
            raise TypeError("rows must be a sequence of mappings or a column mapping")

        if not rows:
            # empty: create empty table from expected schema or raise
            if self._expected_schema is not None:
                return pa.Table.from_arrays([pa.array([], type=f.type) for f in self._expected_schema],
                                            names=[f.name for f in self._expected_schema])
            raise ValueError("rows is empty and no expected_schema was provided")

        # verify all elements are mappings
        if not all(hasattr(r, "keys") for r in rows):
            raise TypeError("every element in rows must be a mapping (e.g., dict)")

        return pa.Table.from_pylist(list(rows))

    def _reconcile_schema(self, table: pa.Table) -> pa.Table:
        """
        If an expected schema was provided, cast columns where possible.
        Missing columns are added as nulls; extra columns are kept.
        """
        if self._expected_schema is None:
            return table

        # ensure all expected columns exist
        for field in self._expected_schema:
            if field.name not in table.column_names:
                # add a null column with the right type
                table = table.append_column(field.name, pa.nulls(len(table), type=field.type))

        # Optionally cast known columns to expected types where feasible
        casts = {}
        for field in self._expected_schema:
            col = table[field.name]
            if not col.type.equals(field.type):
                casts[field.name] = field.type

        if casts:
            table = table.cast(pa.schema(
                [pa.field(name, casts.get(name, table.schema.field(name).type))
                 for name in table.column_names]
            ))

        return table

    # ---------- public API ----------

    def exists(self) -> bool:
        return self._parquet_path.exists()

    def read(self) -> pa.Table:
        if not self.exists():
            raise FileNotFoundError(f"Parquet file not found: {self._parquet_path}")
        return pq.read_table(self._parquet_path)

    def write_new(self, rows: Sequence[Mapping[str, Any]] | Mapping[str, Iterable[Any]]) -> None:
        """
        Overwrite/create the dataset with provided rows.
        """
        table = self._coerce_rows_to_table(rows)
        table = self._reconcile_schema(table)
        self._ensure_parent_dir()
        pq.write_table(table, self._parquet_path)

    def append(self, rows: Sequence[Mapping[str, Any]] | Mapping[str, Iterable[Any]]) -> None:
        """
        Append rows to the Parquet file. For small datasets we read, concat, write.
        For larger datasets, replace with a ParquetWriter (row-group append) approach.
        """
        table = self._coerce_rows_to_table(rows)
        table = self._reconcile_schema(table)
        self._ensure_parent_dir()

        if not self.exists():
            pq.write_table(table, self._parquet_path)
            return

        # Simple append by concat
        old = pq.read_table(self._parquet_path)
        # Make union of schemas (handles new columns)
        new_cols = [c for c in table.column_names if c not in old.column_names]
        for c in new_cols:
            old = old.append_column(c, pa.nulls(len(old), type=table.schema.field(c).type))
        missing_in_new = [c for c in old.column_names if c not in table.column_names]
        for c in missing_in_new:
            table = table.append_column(c, pa.nulls(len(table), type=old.schema.field(c).type))

        combined = pa.concat_tables([old, table], promote=True)
        pq.write_table(combined, self._parquet_path)

    # Convenience: one-shot upsert behavior
    def upsert(self, rows: Sequence[Mapping[str, Any]] | Mapping[str, Iterable[Any]], overwrite: bool = False) -> None:
        if overwrite or not self.exists():
            self.write_new(rows)
        else:
            self.append(rows)
