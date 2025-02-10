import os
from pathlib import Path
from dotenv import load_dotenv
import logging.config
from typing import Dict, Any
import sqlite3

# Load environment variables
load_dotenv('.env.local')

class DatabaseConfig:
    """Database configuration and connection management."""
    
    def __init__(self, db_path: str = "image_generation.db"):
        self.db_path = db_path
        self._connection = None

    @property
    def connection(self):
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None

# API Keys
FAL_KEY = os.getenv('FAL_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# File Paths
INPUT_FILE_PATH = Path(os.getenv('INPUT_FILE_PATH', 'data/inputs/prompts.json'))
OUTPUT_BASE_PATH = Path(os.getenv('OUTPUT_BASE_PATH', 'data/outputs'))

# Ensure directories exist
OUTPUT_BASE_PATH.mkdir(parents=True, exist_ok=True)
INPUT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Database Configuration
DATABASE_CONFIG = {
    "path": "image_generation.db",
    "timeout": 30,
    "max_connections": 5,
    "retry_attempts": 3,
    "retry_delay": 1.0
}

# API Configuration
API_CONFIG = {
    "fal": {
        "url": "https://fal.run/fal-ai/fast-sdxl",
        "timeout": 60,
        "max_retries": 3,
        "retry_delay": 60,
        "batch_size": 3
    },
    "gemini": {
        "timeout": 30,
        "max_retries": 3,
        "retry_delay": 30
    }
}

# API Rate Limits
API_LIMITS = {
    "fal": {
        "max_retries": 3,
        "concurrent_limit": 5,
        "retry_delay": 1.0
    },
    "gemini": {
        "max_retries": 3,
        "concurrent_limit": 3,
        "retry_delay": 1.0
    }
}

# Image Generation Settings
IMAGE_GENERATION = {
    "size": (1024, 1024),
    "negative_prompt": "blurry, low quality, distorted, deformed, ugly, bad anatomy",
    "num_inference_steps": 8,
    "guidance_scale": 7.5,
    "batch_size": 3
}

# Progress Bar Settings
PROGRESS_BAR = {
    "refresh_per_second": 10,
    "transient": True
}

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "rich": {
            "format": "%(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "rich.logging.RichHandler",
            "formatter": "rich",
            "rich_tracebacks": True,
            "tracebacks_show_locals": True,
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": f'generation_log_{os.getenv("RUN_ID", "default")}.log',
            "formatter": "standard",
        },
        "error_file": {
            "class": "logging.FileHandler",
            "filename": "error.log",
            "formatter": "standard",
            "level": "ERROR"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file", "error_file"],
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "propagate": True
        }
    }
}

# Performance Settings
PERFORMANCE = {
    "max_workers": 4,
    "process_pool_workers": 2,
    "thread_pool_workers": 4,
    "chunk_size": 1000
}

# Error Recovery Settings
ERROR_RECOVERY = {
    "max_attempts": 3,
    "base_delay": 1,
    "max_delay": 60,
    "exponential_base": 2
}

# Convenience aliases for backward compatibility
IMAGE_SIZE = IMAGE_GENERATION["size"]
BATCH_SIZE = IMAGE_GENERATION["batch_size"]
DEFAULT_NEGATIVE_PROMPT = IMAGE_GENERATION["negative_prompt"]
DEFAULT_NUM_INFERENCE_STEPS = IMAGE_GENERATION["num_inference_steps"]
DEFAULT_GUIDANCE_SCALE = IMAGE_GENERATION["guidance_scale"]

# Initialize database configuration
db_config = DatabaseConfig(DATABASE_CONFIG["path"])

# Create necessary directories
def ensure_directories():
    """Create all necessary directories."""
    directories = [
        INPUT_FILE_PATH.parent,
        OUTPUT_BASE_PATH,
        OUTPUT_BASE_PATH / "images",
        OUTPUT_BASE_PATH / "evaluations",
        OUTPUT_BASE_PATH / "refined_prompts",
        OUTPUT_BASE_PATH / "logs"
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize
ensure_directories()
logging.config.dictConfig(LOGGING_CONFIG)
