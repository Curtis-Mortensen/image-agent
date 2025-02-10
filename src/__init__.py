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
from pathlib import Path

# Core components
from src.ImageGenerator import ImageGenerator
from src.PromptHandler import PromptHandler
from src.APIClient import FalClient, GeminiClient
from src.ImageVision import ImageVision
from src.PromptRefiner import PromptRefiner
from src.PromptGenerator import PromptGenerator
from src.DatabaseGenerator import DatabaseGenerator, initialize_database
from config import DATABASE_PATH

# Package metadata
__all__ = [
    "ImageGenerator",
    "PromptHandler",
    "FalClient",
    "GeminiClient",
    "ImageVision",
    "PromptRefiner",
    "PromptGenerator",
    "DatabaseGenerator",
    "initialize_database",
    "get_version_info"
]

def get_version_info() -> Dict[str, Any]:
    """Get package version and database information."""
    try:
        generator = DatabaseGenerator()
        db_version = generator.get_version()
            
        return {
            "package_version": __version__,
            "database_version": db_version,
            "author": __author__,
            "license": __license__
        }
    except Exception as e:
        logging.error(f"Error getting version info: {str(e)}")
        return {
            "package_version": __version__,
            "database_version": None,
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
