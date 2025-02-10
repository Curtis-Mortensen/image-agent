import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import sqlite3
from dataclasses import dataclass, asdict
from jsonschema import validate, ValidationError
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class PromptData:
    """Data class for structured prompt data."""
    id: str
    title: str
    scene: str
    mood: str
    prompt: str

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'PromptData':
        return cls(**data)

class PromptHandler:
    """Handles prompt management with SQLite storage."""

    PROMPT_SCHEMA = {
        "type": "object",
        "properties": {
            "prompts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "scene": {"type": "string"},
                        "mood": {"type": "string"},
                        "prompt": {"type": "string"}
                    },
                    "required": ["id", "title", "scene", "mood", "prompt"]
                }
            }
        },
        "required": ["prompts"]
    }

    def __init__(self, input_file_path: Path, output_base_path: Path, 
                 db_path: str = "image_generation.db"):
        self.input_file_path = Path(input_file_path)
        self.output_base_path = Path(output_base_path)
        self.db_path = db_path
        self.prompts_cache: Dict[str, PromptData] = {}
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS prompts (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    scene TEXT NOT NULL,
                    mood TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS iterations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_id TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    image_path TEXT,
                    evaluation_text TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'pending',
                    FOREIGN KEY (prompt_id) REFERENCES prompts(id),
                    UNIQUE(prompt_id, iteration)
                );

                CREATE TABLE IF NOT EXISTS prompt_status (
                    prompt_id TEXT PRIMARY KEY,
                    current_iteration INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (prompt_id) REFERENCES prompts(id)
                );
            """)

    async def setup(self):
        """Create necessary directories."""
        for dir_name in ['images', 'evaluations', 'refined_prompts']:
            (self.output_base_path / dir_name).mkdir(parents=True, exist_ok=True)

    async def load_prompts(self) -> Dict[str, Dict[str, str]]:
        """Load and validate prompts from input file."""
        try:
            if not self.input_file_path.exists():
                raise FileNotFoundError(f"Input file not found: {self.input_file_path}")

            with open(self.input_file_path) as f:
                data = json.load(f)

            validate(instance=data, schema=self.PROMPT_SCHEMA)

            # Store prompts in database
            with sqlite3.connect(self.db_path) as conn:
                for prompt_data in data['prompts']:
                    prompt_obj = PromptData.from_dict(prompt_data)
                    conn.execute("""
                        INSERT OR REPLACE INTO prompts 
                        (id, title, scene, mood, prompt)
                        VALUES (?, ?, ?, ?, ?)
                    """, (prompt_obj.id, prompt_obj.title, prompt_obj.scene,
                          prompt_obj.mood, prompt_obj.prompt))
                    
                    # Initialize status
                    conn.execute("""
                        INSERT OR IGNORE INTO prompt_status (prompt_id)
                        VALUES (?)
                    """, (prompt_obj.id,))
                    
                    self.prompts_cache[prompt_obj.id] = prompt_obj

            return {p.id: asdict(p) for p in self.prompts_cache.values()}

        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            return {}

    async def save_results(self, prompt_id: str, iteration: int,
                          image_path: Optional[Path], prompt: str,
                          evaluation: Optional[Dict[str, Any]] = None) -> None:
        """Save generation results to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO iterations
                    (prompt_id, iteration, image_path, evaluation_text, status)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    prompt_id,
                    iteration,
                    str(image_path) if image_path else None,
                    evaluation['evaluation_text'] if evaluation else None,
                    'completed' if evaluation else 'generated'
                ))

                conn.execute("""
                    UPDATE prompt_status
                    SET current_iteration = ?,
                        status = ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE prompt_id = ?
                """, (
                    iteration,
                    'completed' if evaluation else 'in_progress',
                    prompt_id
                ))

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    async def get_generation_status(self, prompt_id: str) -> Dict[str, Any]:
        """Get current generation status from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT p.*, ps.current_iteration, ps.status, ps.last_updated
                    FROM prompts p
                    JOIN prompt_status ps ON p.id = ps.prompt_id
                    WHERE p.id = ?
                """, (prompt_id,))
                
                result = cursor.fetchone()
                if not result:
                    return {
                        "prompt_id": prompt_id,
                        "status": "not_found",
                        "iterations_completed": 0
                    }

                return {
                    "prompt_id": prompt_id,
                    "status": result['status'],
                    "iterations_completed": result['current_iteration'],
                    "last_updated": result['last_updated'],
                    "prompt_data": asdict(self.prompts_cache.get(prompt_id))
                }

        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            return {"prompt_id": prompt_id, "status": "error", "error": str(e)}

    async def cleanup(self):
        """Cleanup resources."""
        self.prompts_cache.clear()

    async def __aenter__(self):
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
