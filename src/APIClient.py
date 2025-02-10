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

    def _init_db(self) -> None:
        """Initialize the database synchronously."""
        # Database initialization is now handled by DatabaseGenerator
        pass

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
        try:
            logger.debug(f"Received update of type: {type(update)}")
            logger.debug(f"Update object attributes: {dir(update)}")
            
            if isinstance(update, fal_client.InProgress):
                # Handle progress updates
                if hasattr(update, 'logs') and update.logs:
                    for log in update.logs:
                        logger.debug(f"Generation progress: {log['message']}")
            elif isinstance(update, fal_client.Completed):
                # Handle completion
                logger.info("Generation completed successfully")
                logger.debug(f"Completed update full structure: {update.__dict__}")
            elif isinstance(update, fal_client.Failed):
                # Handle errors
                error_msg = getattr(update, 'error', 'Unknown error')
                logger.error(f"Generation failed: {error_msg}")
                raise fal_client.APIError(str(error_msg))
            elif isinstance(update, fal_client.Queued):
                # Handle queued status
                logger.debug("Request queued")
            else:
                logger.warning(f"Unknown status update type: {type(update)}")
        except Exception as e:
            logger.error(f"Error handling status update: {str(e)}")
            raise

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

                try:
                    # Create a future to handle the async completion
                    future = asyncio.Future()
                    
                    def handle_update(update):
                        try:
                            self._handle_status_update(update)
                            if isinstance(update, fal_client.Completed):
                                if not future.done():
                                    logger.debug(f"Completed update full structure: {update.__dict__}")
                                    logger.debug(f"Update object dir: {dir(update)}")
                                    # Try to access the images attribute directly
                                    if hasattr(update, 'images'):
                                        result_dict = {
                                            "images": [{
                                                "url": update.images[0].url if update.images else None
                                            }]
                                        }
                                        future.set_result(result_dict)
                                    else:
                                        logger.error("Update object has no 'images' attribute")
                                        future.set_exception(KeyError("No 'images' in update object"))
                            elif isinstance(update, fal_client.Failed):
                                if not future.done():
                                    future.set_exception(fal_client.APIError(str(update.error)))
                        except Exception as e:
                            logger.error(f"Error in handle_update: {str(e)}, update type: {type(update)}")
                            if not future.done():
                                future.set_exception(e)

                    # Start the generation process
                    fal_client.subscribe(
                        "fal-ai/fast-lightning-sdxl",
                        arguments=arguments,
                        with_logs=True,
                        on_queue_update=handle_update
                    )
                    
                    # Wait for the result with a timeout
                    try:
                        result = await asyncio.wait_for(future, timeout=60.0)
                    except asyncio.TimeoutError:
                        logger.error("Generation timed out after 60 seconds")
                        raise
                    
                    if not result:
                        logger.error("Empty response from fal.ai API")
                        return None
                        
                    # Validate response structure
                    if not isinstance(result, dict):
                        logger.error(f"Unexpected response type: {type(result)}")
                        return None
                        
                    if 'images' not in result or not result['images']:
                        logger.error("No images in response")
                        return None
                        
                    first_image = result['images'][0]
                    if 'url' not in first_image:
                        logger.error("No URL in image data")
                        return None
                        
                    # Log success and return the result
                    logger.info(f"Successfully generated image: {first_image['url']}")
                    await self.call_tracker.log_call(
                        "fal.ai", 
                        "generate_image",
                        "success"
                    )
                    return result
                    
                except fal_client.AuthenticationError:
                    logger.error("Invalid FAL API key")
                    raise
                except fal_client.RequestError as e:
                    logger.error(f"Invalid request parameters: {str(e)}")
                    raise
                except fal_client.APIError as e:
                    logger.error(f"FAL API error: {str(e)}")
                    raise
                except KeyError as e:
                    logger.error(f"Unexpected response format: missing key {str(e)}")
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error: {str(e)}")
                    raise

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
