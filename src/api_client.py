import aiohttp
import google.generativeai as genai
import logging
import json
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path
from PIL import Image
import backoff
from aiohttp import ClientTimeout
from tenacity import retry, stop_after_attempt, wait_exponential
import fal_client
import sqlite3
from datetime import datetime

from config import (
    DEFAULT_NUM_INFERENCE_STEPS, 
    DEFAULT_NEGATIVE_PROMPT, 
    DEFAULT_GUIDANCE_SCALE, 
    IMAGE_SIZE,
    DATABASE_PATH
)

logger = logging.getLogger(__name__)

class APICallTracker:
    """Tracks API calls and rate limits."""
    
    def __init__(self, db_path: str = DATABASE_PATH) -> None:
        self.db_path = db_path
        self._init_db()  # Initialize DB on creation

    def _init_db(self) -> None:
        """Initialize the database synchronously."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    api_name TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    async def log_call(
        self,
        api_name: str,
        endpoint: str,
        status: str = "success",
        error: Optional[str] = None
    ) -> None:
        """Log an API call to the database.

        Args:
            api_name: Name of the API service
            endpoint: Specific endpoint called
            status: Status of the call (success/error)
            error: Error message if status is error
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO api_calls (api_name, endpoint, status, error)
                VALUES (?, ?, ?, ?)
                """,
                (api_name, endpoint, status, error)
            )

class FalClient:
    """Client for interacting with the fal.ai API with enhanced tracking."""

    def __init__(
        self,
        api_key: str,
        timeout: int = 60,
        db_path: str = DATABASE_PATH
    ) -> None:
        self.api_key = api_key
        self.timeout = ClientTimeout(total=timeout)
        self.call_tracker = APICallTracker(db_path)  # This initializes DB
        self.session: Optional[aiohttp.ClientSession] = None
        fal_client.api_key = api_key
        
        # Add rate limiting parameters
        self.max_retries = 3
        self.retry_delay = 1.0
        self.concurrent_limit = asyncio.Semaphore(5)  # Limit concurrent requests

    async def setup(self) -> None:
        """Initialize resources."""
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=self.timeout)

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None

    def _handle_status_update(self, update: Any) -> None:
        """Handle status updates from FAL.ai.
        
        Args:
            update: Status update from FAL.ai
        """
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                logger.debug(f"Generation progress: {log['message']}")

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3
    )
    async def generate_image(self, prompt: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate an image with enhanced error handling and tracking."""
        async with self.concurrent_limit:  # Rate limiting
            try:
                arguments = {
                    "prompt": prompt,
                    "negative_prompt": kwargs.get('negative_prompt', DEFAULT_NEGATIVE_PROMPT),
                    "num_inference_steps": kwargs.get('num_inference_steps', DEFAULT_NUM_INFERENCE_STEPS),
                    "guidance_scale": kwargs.get('guidance_scale', DEFAULT_GUIDANCE_SCALE),
                    "width": kwargs.get('width', IMAGE_SIZE[0]),
                    "height": kwargs.get('height', IMAGE_SIZE[1]),
                }

                logger.debug(f"Generating image with parameters: {arguments}")

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: fal_client.subscribe(
                        "fal-ai/fast-lightning-sdxl",
                        arguments=arguments,
                        with_logs=True,
                        on_queue_update=self._handle_status_update
                    )
                )

                await self.call_tracker.log_call(
                    "fal.ai", 
                    "generate_image",
                    "success"
                )
                return result

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Image generation error: {error_msg}")
                await self.call_tracker.log_call(
                    "fal.ai",
                    "generate_image",
                    "error",
                    error_msg
                )
                raise

class GeminiClient:
    """Client for interacting with Google Gemini API with enhanced features."""

    def __init__(self, api_key: str, db_path: str = DATABASE_PATH):
        self.api_key = api_key
        self.db_path = db_path
        self.tracker = APICallTracker(db_path)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        genai.configure(api_key=api_key)
        
        # Add rate limiting
        self.concurrent_limit = asyncio.Semaphore(3)

    async def evaluate_image(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        """Evaluate image with rate limiting and tracking."""
        async with self.concurrent_limit:
            try:
                response = self.model.generate_content([
                    "Describe this image in detail, focusing on visual elements, "
                    "composition, and overall scene.",
                    image
                ])
                response.resolve()

                if response.text:
                    await self.tracker.log_call("gemini", "evaluate_image", "success")
                    return {"evaluation_text": response.text}
                return None

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Image evaluation error: {error_msg}")
                await self.tracker.log_call(
                    "gemini",
                    "evaluate_image",
                    "error",
                    error_msg
                )
                return None

    async def refine_prompt(self, prompt: str, evaluation: Dict[str, Any]) -> Optional[str]:
        """Refine prompt with rate limiting and tracking."""
        async with self.concurrent_limit:
            try:
                response = self.model.generate_content([
                    f"Original prompt: {prompt}\n"
                    f"Evaluation: {evaluation.get('evaluation_text', '')}\n"
                    "Please suggest improvements to the prompt."
                ])
                response.resolve()

                if response.text:
                    await self.tracker.log_call("gemini", "refine_prompt", "success")
                    return response.text
                return None

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Prompt refinement error: {error_msg}")
                await self.tracker.log_call(
                    "gemini",
                    "refine_prompt",
                    "error",
                    error_msg
                )
                return None
