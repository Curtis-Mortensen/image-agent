"""
Configuration settings for the image generation pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import logging.config
from typing import Dict, Any
import sqlite3

# Load environment variables
load_dotenv()

# API Keys and Authentication
FAL_KEY = os.getenv("FAL_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# File Paths
BASE_DIR = Path(__file__).parent
INPUT_FILE_PATH = BASE_DIR / "data/inputs/prompts.json"
OUTPUT_BASE_PATH = BASE_DIR / "data/outputs"
DATABASE_PATH = BASE_DIR / "data/database/image_generation.db"

# Pipeline Configuration
PIPELINE_CONFIG = {
    "max_iterations": 3,
    "batch_size": {
        "min": 1,
        "max": 10,
        "default": 5
    },
    "quality_threshold": 0.7,  # Minimum score for a variant to be considered good
    "refinement_threshold": 0.85,  # Score above which refinement is not needed
}

# Image Generation Settings
IMAGE_GENERATION = {
    "size": {
        "width": 1024,
        "height": 1024
    },
    "negative_prompt": "blurry, low quality, distorted, deformed, ugly, bad anatomy",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "model": {
        "default": "fal-ai/fast-lightning-sdxl",
        "alternatives": [
            "stabilityai/stable-diffusion-xl-base-1.0",
            "runwayml/stable-diffusion-v1-5"
        ]
    }
}

# Database Settings
DATABASE_CONFIG = {
    "version": "1.0.0",
    "tables": {
        "generated_images": {
            "columns": [
                "id INTEGER PRIMARY KEY AUTOINCREMENT",
                "prompt_id TEXT",
                "iteration INTEGER",
                "variant INTEGER",
                "image_path TEXT UNIQUE",
                "prompt_text TEXT",
                "evaluation_text TEXT",
                "evaluation_score REAL",
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ],
            "indices": [
                "CREATE INDEX IF NOT EXISTS idx_prompt_iter ON generated_images(prompt_id, iteration)",
                "CREATE INDEX IF NOT EXISTS idx_evaluation_score ON generated_images(evaluation_score)"
            ]
        },
        "best_images": {
            "columns": [
                "prompt_id TEXT",
                "iteration INTEGER",
                "best_image_id INTEGER",
                "evaluation_score REAL",
                "FOREIGN KEY(best_image_id) REFERENCES generated_images(id)",
                "PRIMARY KEY(prompt_id, iteration)"
            ]
        }
    }
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
BATCH_SIZE = PIPELINE_CONFIG["batch_size"]["default"]  # Default batch size
DEFAULT_NEGATIVE_PROMPT = IMAGE_GENERATION["negative_prompt"]
DEFAULT_NUM_INFERENCE_STEPS = IMAGE_GENERATION["num_inference_steps"]
DEFAULT_GUIDANCE_SCALE = IMAGE_GENERATION["guidance_scale"]

# Multi-Variant Pipeline Settings
PIPELINE_CONFIG = {
    "max_iterations": 3,
    "batch_size": 5,  # Number of variants per iteration
    "quality_threshold": 0.7,  # Minimum score for a variant to be considered good
    "refinement_threshold": 0.85,  # Score above which refinement is not needed
    "naming": {
        "image_format": "{prompt_id}_iter{iteration}_v{variant}.png",
        "symlink_format": "current/{prompt_id}.png"
    }
}

# Update Image Generation settings to align with pipeline
IMAGE_GENERATION.update({
    "batch_size": PIPELINE_CONFIG["batch_size"]  # Ensure consistency
})

# Add new convenience aliases
MAX_ITERATIONS = PIPELINE_CONFIG["max_iterations"]
BATCH_SIZE = PIPELINE_CONFIG["batch_size"]  # Override the previous BATCH_SIZE
QUALITY_THRESHOLD = PIPELINE_CONFIG["quality_threshold"]
REFINEMENT_THRESHOLD = PIPELINE_CONFIG["refinement_threshold"]

# Initialize logging
logging.config.dictConfig(LOGGING_CONFIG)
