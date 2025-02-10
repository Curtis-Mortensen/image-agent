import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional
import sqlite3

logger = logging.getLogger(__name__)

class DatabaseUtils:
    @staticmethod
    async def execute_query(db_path: str, query: str, params: tuple = None) -> Optional[Any]:
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params or ())
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            return None

    @staticmethod
    async def get_status(db_path: str, entity_id: str, table: str) -> Dict[str, Any]:
        query = f"SELECT status, last_updated FROM {table} WHERE id = ?"
        result = await DatabaseUtils.execute_query(db_path, query, (entity_id,))
        return dict(result[0]) if result else {"status": "unknown", "last_updated": None}

class PathManager:
    @staticmethod
    def ensure_paths(base_path: Path, subdirs: list) -> Dict[str, Path]:
        paths = {}
        for subdir in subdirs:
            path = base_path / subdir
            path.mkdir(parents=True, exist_ok=True)
            paths[subdir] = path
        return paths
