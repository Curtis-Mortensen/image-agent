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

logger = logging.getLogger(__name__)

class FalClient:
    """Client for interacting with the fal.ai API."""
    
    def __init__(self, api_key: str, timeout: int = 60):
        self.api_key = api_key
        self.timeout = timeout
        fal_client.api_key = api_key

    async def setup(self):
        """Initialize the client."""
        pass  # No setup needed for new client

    async def cleanup(self):
        """Clean up resources."""
        pass  # No cleanup needed for new client

    def _handle_status_update(self, update):
        """Handle status updates from FAL.ai."""
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                logger.debug(f"Generation progress: {log['message']}")

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3
    )
    async def generate_image(self, prompt: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Generate an image using fal.ai API with automatic retries.
        
        Args:
            prompt: The text prompt for image generation
            **kwargs: Additional parameters for image generation
            
        Returns:
            Dictionary containing the API response or None if failed
        """
        try:
            # Prepare generation parameters
            arguments = {
                "prompt": prompt,
                "negative_prompt": kwargs.get('negative_prompt', "blurry, low quality, distorted, deformed"),
                "num_inference_steps": kwargs.get('num_inference_steps', 30),
                "guidance_scale": kwargs.get('guidance_scale', 7.5),
                "width": kwargs.get('width', 1024),
                "height": kwargs.get('height', 1024),
            }

            # Use asyncio to run the synchronous fal_client in a thread pool
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

            return result

        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            raise

[Rest of the file remains unchanged...]
