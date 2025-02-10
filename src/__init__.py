"""
AI Image Generation Pipeline
A system for batch processing and refinement of AI-generated images using SQLite storage.

This package provides a complete pipeline for:
- Loading and managing prompts
- Generating images using FAL.ai
- Evaluating images using Google Gemini
- Refining prompts based on evaluations
- Tracking all operations in SQLite database
"""

__version__ = "0.1.0"
__author__ = "Curtis Mortensen"
__license__ = "MIT"

import logging
from typing import Dict, Any
import sqlite3
from pathlib import Path

# Core components
from src.image_generator import ImageGenerator
from src.prompt_handler import PromptHandler
from src.api_client import FalClient, GeminiClient
from src.image_evaluator import ImageEvaluator
from src.prompt_refiner import PromptRefiner
from src.main import ImageGenerationPipeline
from config import DATABASE_PATH

# Package metadata
__all__ = [
    "ImageGenerator",
    "PromptHandler",
    "FalClient",
    "GeminiClient",
    "ImageEvaluator",
    "PromptRefiner",
    "ImageGenerationPipeline",
    "initialize_database",
    "get_version_info"
]

def initialize_database(db_path: str = DATABASE_PATH) -> None:
    """Initialize the SQLite database with all required tables."""
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            -- Version tracking
            CREATE TABLE IF NOT EXISTS version_info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                installed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Insert current version
        conn.execute("INSERT OR IGNORE INTO version_info (version) VALUES (?)", 
                    (__version__,))

def get_version_info() -> Dict[str, Any]:
    """Get package version and database information."""
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.execute("""
                SELECT version, installed_at 
                FROM version_info 
                ORDER BY installed_at DESC 
                LIMIT 1
            """)
            db_version = cursor.fetchone()
            
            return {
                "package_version": __version__,
                "database_version": db_version[0] if db_version else None,
                "installed_at": db_version[1] if db_version else None,
                "author": __author__,
                "license": __license__
            }
    except Exception as e:
        logging.error(f"Error getting version info: {str(e)}")
        return {
            "package_version": __version__,
            "database_version": None,
            "installed_at": None,
            "author": __author__,
            "license": __license__
        }

# Initialize logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Optional: Initialize database on import
try:
    initialize_database()
except Exception as e:
    logging.warning(f"Could not initialize database: {str(e)}")