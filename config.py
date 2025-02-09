import os
from pathlib import Path
from dotenv import load_dotenv
import logging.config
from typing import Dict, Any

# Load environment variables from .env.local
load_dotenv('.env.local')

# API Keys
FAL_KEY = os.getenv('FAL_KEY')
if not FAL_KEY:
    raise ValueError("FAL_KEY not found in .env.local")

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env.local")

# File Paths
PROJECT_ROOT = Path(__file__).parent
INPUT_FILE_PATH = PROJECT_ROOT / "data" / "inputs" / "prompts.json"
OUTPUT_BASE_PATH = PROJECT_ROOT / "data" / "outputs"

# API Configuration
FAL_API_URL = "https://fal.run/fal-ai/fast-sdxl"
FAL_API_TIMEOUT = 60  # seconds
FAL_MAX_RETRIES = 3
FAL_RETRY_DELAY = 60  # seconds

# Image Generation Settings
IMAGE_SIZE = (1024, 1024)
BATCH_SIZE = 3
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, deformed, ugly, bad anatomy"
DEFAULT_NUM_INFERENCE_STEPS = 8 # Updated to a valid value: 8
DEFAULT_GUIDANCE_SCALE = 7.5

# Rich Progress Bar Settings
PROGRESS_REFRESH_PER_SECOND = 10
PROGRESS_TRANSIENT = True

# Logging Configuration
LOGGING_CONFIG: Dict[str, Any] = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'rich': {
            'format': '%(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'rich.logging.RichHandler',
            'formatter': 'rich',
            'rich_tracebacks': True,
            'tracebacks_show_locals': True,
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': f'generation_log_{os.getenv("RUN_ID", "default")}.log',
            'formatter': 'standard',
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file'],
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'propagate': True
        }
    }
}

# Thread/Process Pool Settings
MAX_WORKERS = 4
PROCESS_POOL_WORKERS = 2

# API Retry Settings
RETRY_MULTIPLIER = 1
RETRY_MIN_SECONDS = 4
RETRY_MAX_SECONDS = 10
MAX_RETRY_ATTEMPTS = 3

# Create necessary directories
INPUT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_BASE_PATH.mkdir(parents=True, exist_ok=True)

# Initialize logging
logging.config.dictConfig(LOGGING_CONFIG)
