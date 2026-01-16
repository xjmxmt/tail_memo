from loguru import logger
import threading
import uuid
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor


class PostgreSQLManager:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "mydatabase",
        user: str = "postgres",
        password: str = "",
    ):
        self.connection_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }
        self.connection = psycopg2.connect(**self.connection_params)
        self.connection.autocommit = False
        self._lock = threading.Lock()
        self._migrate_history_table()
        self._create_history_table()

    def _migrate_history_table(self) -> None:
        """
        If a pre-existing history table had the old group-chat columns,
        rename it, create the new schema, copy the intersecting file, then
        drop the old table.
        """
        with self._lock:
            cursor = self.connection.cursor()
            try:
                # Check if history table exists
                cursor.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'history'
                    )
                    """
                )
                table_exists = cursor.fetchone()[0]

                if not table_exists:
                    self.connection.commit()
                    return  # nothing to migrate

                # Get existing columns
                cursor.execute(
                    """
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                    AND table_name = 'history'
                    """
                )
                old_cols = {row[0] for row in cursor.fetchall()}

                expected_cols = {
                    "id",
                    "memory_id",
                    "old_memory",
                    "new_memory",
                    "event",
                    "created_at",
                    "updated_at",
                    "is_deleted",
                    "actor_id",
                    "role",
                }

                if old_cols == expected_cols:
                    self.connection.commit()
                    return

                logger.info("Migrating history table to new schema (no convo columns).")

                # Clean up any existing history_old table from previous failed migration
                cursor.execute("DROP TABLE IF EXISTS history_old")

                # Rename the current history table
                cursor.execute("ALTER TABLE history RENAME TO history_old")

                # Create the new history table with updated schema
                cursor.execute(
                    """
                    CREATE TABLE history (
                        id           TEXT PRIMARY KEY,
                        memory_id    TEXT,
                        old_memory   TEXT,
                        new_memory   TEXT,
                        event        TEXT,
                        created_at   TIMESTAMP,
                        updated_at   TIMESTAMP,
                        is_deleted   INTEGER,
                        actor_id     TEXT,
                        role         TEXT
                    )
                    """
                )

                # Copy file from old table to new table
                intersecting = list(expected_cols & old_cols)
                if intersecting:
                    cols_csv = ", ".join(intersecting)
                    cursor.execute(
                        f"INSERT INTO history ({cols_csv}) SELECT {cols_csv} FROM history_old"
                    )

                # Drop the old table
                cursor.execute("DROP TABLE history_old")

                # Commit the transaction
                self.connection.commit()
                logger.info("History table migration completed successfully.")

            except Exception as e:
                # Rollback the transaction on any error
                self.connection.rollback()
                logger.error(f"History table migration failed: {e}")
                raise
            finally:
                cursor.close()

    def _create_history_table(self) -> None:
        with self._lock:
            cursor = self.connection.cursor()
            try:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS history (
                        id           TEXT PRIMARY KEY,
                        memory_id    TEXT,
                        old_memory   TEXT,
                        new_memory   TEXT,
                        event        TEXT,
                        created_at   TIMESTAMP,
                        updated_at   TIMESTAMP,
                        is_deleted   INTEGER,
                        actor_id     TEXT,
                        role         TEXT
                    )
                    """
                )
                self.connection.commit()
            except Exception as e:
                self.connection.rollback()
                logger.error(f"Failed to create history table: {e}")
                raise
            finally:
                cursor.close()

    def add_history(
        self,
        memory_id: str,
        old_memory: Optional[str],
        new_memory: Optional[str],
        event: str,
        *,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        is_deleted: int = 0,
        actor_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> None:
        with self._lock:
            cursor = self.connection.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO history (
                        id, memory_id, old_memory, new_memory, event,
                        created_at, updated_at, is_deleted, actor_id, role
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(uuid.uuid4()),
                        memory_id,
                        old_memory,
                        new_memory,
                        event,
                        created_at,
                        updated_at,
                        is_deleted,
                        actor_id,
                        role,
                    ),
                )
                self.connection.commit()
            except Exception as e:
                self.connection.rollback()
                logger.error(f"Failed to add history record: {e}")
                raise
            finally:
                cursor.close()

    def get_history(self, memory_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            cursor = self.connection.cursor()
            try:
                cursor.execute(
                    """
                    SELECT id, memory_id, old_memory, new_memory, event,
                           created_at, updated_at, is_deleted, actor_id, role
                    FROM history
                    WHERE memory_id = %s
                    ORDER BY created_at ASC, updated_at ASC
                    """,
                    (memory_id,),
                )
                rows = cursor.fetchall()
            finally:
                cursor.close()

        return [
            {
                "id": r[0],
                "memory_id": r[1],
                "old_memory": r[2],
                "new_memory": r[3],
                "event": r[4],
                "created_at": r[5].isoformat() if r[5] else None,
                "updated_at": r[6].isoformat() if r[6] else None,
                "is_deleted": bool(r[7]),
                "actor_id": r[8],
                "role": r[9],
            }
            for r in rows
        ]

    def reset(self) -> None:
        """Drop and recreate the history table."""
        with self._lock:
            cursor = self.connection.cursor()
            try:
                cursor.execute("DROP TABLE IF EXISTS history")
                self.connection.commit()
            except Exception as e:
                self.connection.rollback()
                logger.error(f"Failed to reset history table: {e}")
                raise
            finally:
                cursor.close()
        self._create_history_table()

    def close(self) -> None:
        if self.connection:
            self.connection.close()
            self.connection = None

    def __del__(self):
        self.close()
