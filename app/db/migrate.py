"""Additive schema migrations.

Phase 2 adds ``documents.document_key`` for cross-version grouping. The project
does not use Alembic yet, so this module applies idempotent, additive-only
changes for both SQLite and PostgreSQL. It never drops or rewrites data.
"""

from __future__ import annotations

import logging

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

_ADDITIVE_COLUMNS: dict[str, dict[str, str]] = {
    "documents": {
        "document_key": "VARCHAR(128)",
    },
}


async def apply_additive_migrations(engine: AsyncEngine) -> list[str]:
    """Add missing nullable columns. Returns the list of applied changes."""

    applied: list[str] = []

    def _inspect(sync_connection) -> dict[str, set[str]]:
        inspector = inspect(sync_connection)
        tables = set(inspector.get_table_names())
        return {
            table: {column["name"] for column in inspector.get_columns(table)}
            for table in tables
        }

    async with engine.begin() as connection:
        existing = await connection.run_sync(_inspect)

        for table, columns in _ADDITIVE_COLUMNS.items():
            present = existing.get(table)
            if present is None:
                continue
            for column, ddl_type in columns.items():
                if column in present:
                    continue
                await connection.execute(
                    text(f'ALTER TABLE {table} ADD COLUMN "{column}" {ddl_type}')
                )
                applied.append(f"{table}.{column}")

        if "documents" in existing:
            # CREATE INDEX IF NOT EXISTS is supported by SQLite 3.3+ and PostgreSQL 9.5+.
            await connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_documents_document_key "
                    'ON documents ("document_key")'
                )
            )

        # Create the benchmark_runs table if it does not exist yet (Phase 4).
        if "benchmark_runs" not in existing:
            await connection.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS benchmark_runs ("
                    "id VARCHAR(64) PRIMARY KEY, "
                    "status VARCHAR(32) NOT NULL DEFAULT 'pending', "
                    "created_at TIMESTAMP NOT NULL, "
                    "updated_at TIMESTAMP NOT NULL, "
                    "total_questions INTEGER NOT NULL DEFAULT 0, "
                    "completed_questions INTEGER NOT NULL DEFAULT 0, "
                    "results_json TEXT, "
                    "summary_json TEXT, "
                    "error TEXT"
                    ")"
                )
            )
            applied.append("benchmark_runs (table)")

    if applied:
        logger.info("Applied additive migrations: %s", ", ".join(applied))
    return applied
