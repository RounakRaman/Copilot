"""
Database layer for persisting the user profile in Postgres (Neon).

This replaces the file-based profile.json storage. The previous approach
broke twice for two different reasons:
1. Writing into the read-only app source directory -> PermissionError.
2. Writing into /tmp -> works, but /tmp is wiped on every restart/redeploy,
   so "permanent" storage was never actually permanent.

A real database, on a host that doesn't expire/pause your data, is the
actual fix for "saved forever." This module talks to Neon Postgres via
a single DATABASE_URL connection string.

Setup:
    1. Create a free project at https://neon.tech
    2. Copy the connection string from the Neon dashboard (it looks like
       postgresql://user:password@host/dbname?sslmode=require)
    3. On Render: Dashboard -> your service -> Environment -> add:
           DATABASE_URL = <that connection string>
    4. That's it -- this module creates the table itself on first use.

If DATABASE_URL is not set, this module raises a clear error rather than
silently falling back to file storage, so you always know which mode
you're actually running in.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import psycopg2
import psycopg2.extras

DATABASE_URL = os.environ.get("DATABASE_URL")

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS profile (
    id INTEGER PRIMARY KEY DEFAULT 1,
    data JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT single_row CHECK (id = 1)
);
"""
# This app only ever stores one profile (single user), so we pin the row
# to id=1 and use upsert semantics. If you ever need multi-user support,
# this table needs a real user_id column and auth -- that's a bigger
# change, not a tweak.


def get_connection():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it as an environment variable "
            "pointing at your Neon Postgres connection string."
        )
    return psycopg2.connect(DATABASE_URL)


def init_db() -> None:
    """Create the profile table if it doesn't already exist. Safe to call repeatedly."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(_TABLE_DDL)
        conn.commit()
    finally:
        conn.close()


def load_profile_dict() -> Optional[dict]:
    """Return the stored profile as a dict, or None if nothing is saved yet."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM profile WHERE id = 1;")
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def save_profile_dict(data: dict) -> None:
    """Upsert the single profile row."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO profile (id, data, updated_at)
                VALUES (1, %s, now())
                ON CONFLICT (id) DO UPDATE
                SET data = EXCLUDED.data, updated_at = now();
                """,
                [psycopg2.extras.Json(data)],
            )
        conn.commit()
    finally:
        conn.close()
