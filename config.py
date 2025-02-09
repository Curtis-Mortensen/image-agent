import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv('.env.local')

# API Keys
FAL_AI_API_KEY = os.getenv('FAL_AI_API_KEY')
if not FAL_AI_API_KEY:
    raise ValueError("FAL_AI_API_KEY not found in .env.local")

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env.local")

# File Paths
PROJECT_ROOT = Path(__file__).parent
INPUT_FILE_PATH = PROJECT_ROOT / "data" / "inputs" / "prompts.json"
OUTPUT_BASE_PATH = PROJECT_ROOT / "data" / "outputs"

# API Configuration
FAL_API_URL = "https://110602490-fast-sdxl.fal.run"
FAL_API_TIMEOUT = 60  # seconds
FAL_MAX_RETRIES = 3
FAL_RETRY_DELAY = 60  # seconds

# Image Generation Settings
IMAGE_SIZE = (1024, 1024)
BATCH_SIZE = 3
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, deformed, ugly, bad anatomy"
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Create necessary directories
INPUT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_BASE_PATH.mkdir(parents=True, exist_ok=True)
