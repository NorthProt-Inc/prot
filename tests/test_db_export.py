"""Tests for DB CSV export."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from prot.db import export_tables, EXPORT_TABLES


@pytest.mark.asyncio
class TestExportTables:
    async def test_export_creates_csv_per_table(self, tmp_path):
        mock_conn = AsyncMock()

        async def fake_copy(query, *, output, **kw):
            Path(output).write_text("id,name\n")
            return "COPY 0"

        mock_conn.copy_from_query = fake_copy
        mock_pool = MagicMock()

        @asynccontextmanager
        async def fake_acquire():
            yield mock_conn

        mock_pool.acquire = fake_acquire

        await export_tables(mock_pool, str(tmp_path))
        for name in EXPORT_TABLES:
            assert (tmp_path / f"{name}.csv").exists()

    async def test_export_creates_output_dir(self, tmp_path):
        out = tmp_path / "nested" / "dir"
        mock_conn = AsyncMock()

        async def fake_copy(query, *, output, **kw):
            Path(output).write_text("")
            return "COPY 0"

        mock_conn.copy_from_query = fake_copy
        mock_pool = MagicMock()

        @asynccontextmanager
        async def fake_acquire():
            yield mock_conn

        mock_pool.acquire = fake_acquire

        await export_tables(mock_pool, str(out))
        assert out.exists()

    async def test_export_continues_on_single_table_failure(self, tmp_path):
        call_count = 0

        async def failing_copy(query, *, output, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("simulated failure")
            Path(output).write_text("data")
            return "COPY 1"

        mock_conn = AsyncMock()
        mock_conn.copy_from_query = failing_copy
        mock_pool = MagicMock()

        @asynccontextmanager
        async def fake_acquire():
            yield mock_conn

        mock_pool.acquire = fake_acquire

        await export_tables(mock_pool, str(tmp_path))
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == 4  # 5 tables, 1 failed
