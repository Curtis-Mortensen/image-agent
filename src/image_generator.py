import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from PIL import Image
import fal_client
from datetime import datetime
import io
import aiohttp  # Import aiohttp for downloading

logger = logging.getLogger(__name__)

class ImageGenerator:
    """Handles the complete image generation process."""

    def __init__(self, fal_api_key: str, gemini_api_key: str, output_base_path: Path,
                 batch_size: int = 3):
        """Initialize the image generator with API clients."""
        self.output_base_path = Path(output_base_path)
        self.batch_size = batch_size

        # Initialize API clients
        fal_client.api_key = fal_api_key
        from src.api_client import FalClient, GeminiClient
        self.fal_client = FalClient(fal_api_key)
        self.gemini_client = GeminiClient(gemini_api_key)

    async def setup(self):
        """Initialize resources."""
        await self.fal_client.setup()

    async def cleanup(self):
        """Cleanup resources."""
        await self.fal_client.cleanup()

    def _construct_full_prompt(self, prompt_data: Dict) -> str:
        """Construct the full prompt from prompt data."""
        return (
            f"Title: {prompt_data['title']}\n"
            f"Scene: {prompt_data['scene']}\n"
            f"Mood: {prompt_data['mood']}\n"
            f"Prompt: {prompt_data['prompt']}"
        )

    async def generate_and_evaluate(self, prompt_data: Dict, prompt_id: str,
                                  progress_callback=None) -> Tuple[Optional[Path], Optional[Dict]]:
        """
        Complete generation process including evaluation and refinement.

        Args:
            prompt_data: Dictionary containing prompt information
            prompt_id: Unique identifier for the prompt
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (final_image_path, evaluation_results)
        """
        try:
            # Initial generation
            if progress_callback:
                progress_callback(f"Generating initial image for {prompt_id}")

            full_prompt = self._construct_full_prompt(prompt_data)
            initial_image = await self.generate_image(
                full_prompt,
                prompt_id,
                iteration=1,
                model_id="fal-ai/fast-lightning-sdxl"
            )

            if not initial_image:
                return None, None

            # Evaluate image
            if progress_callback:
                progress_callback(f"Evaluating image for {prompt_id}")

            evaluation = await self.evaluate_image(initial_image)
            if not evaluation:
                return initial_image, None

            # Refine prompt
            if progress_callback:
                progress_callback(f"Refining prompt for {prompt_id}")

            refined_prompt = await self.refine_prompt(full_prompt, evaluation)
            if not refined_prompt:
                return initial_image, evaluation

            # Generate refined version
            if progress_callback:
                progress_callback(f"Generating refined image for {prompt_id}")

            refined_image = await self.generate_image(
                refined_prompt,
                prompt_id,
                iteration=2,
                model_id="fal-ai/fast-lightning-sdxl"
            )

            return refined_image or initial_image, evaluation

        except Exception as e:
            logger.error(f"Error in generation process: {str(e)}")
            return None, None

    async def generate_image(self, prompt: str, prompt_id: str,
                           iteration: int = 1, **kwargs) -> Optional[Path]:
        """Generate a single image."""
        try:
            result = await self.fal_client.generate_image(prompt, **kwargs)
            if not result:
                return None

            # Save the image
            output_dir = self.output_base_path / prompt_id
            output_dir.mkdir(parents=True, exist_ok=True)
            image_path = output_dir / f"iteration_{iteration}.png"

            # Save image data
            if 'images' in result and result['images']:
                image_info = result['images'][0]  # Assuming first image is an Image object
                image_url = image_info.get('url') # Get image URL from Image object

                if image_url:
                    logger.debug(f"Image URL found: {image_url}")
                    try:
                        async with aiohttp.ClientSession() as session: # Create aiohttp session
                            async with session.get(image_url) as resp: # Download image data from URL
                                if resp.status == 200:
                                    image_bytes = await resp.read() # Read image bytes
                                    pil_image = Image.open(io.BytesIO(image_bytes)) # Open image with PIL
                                    pil_image.save(image_path) # Save image to file
                                    logger.debug(f"Saving image to path: {image_path}")
                                    return image_path
                                else:
                                    logger.error(f"Failed to download image from {image_url}, status: {resp.status}")
                                    return None
                    except Exception as download_error:
                        logger.error(f"Error downloading or saving image: {download_error}")
                        return None
                else:
                    logger.error("Image URL not found in API response")
                    return None

            return None

        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            return None

    async def evaluate_image(self, image_path: Path) -> Optional[Dict]:
        """Evaluate an image using Gemini Vision."""
        try:
            logger.debug(f"Evaluating image at path: {image_path}")
            return await self.gemini_client.evaluate_image(Image.open(image_path))
        except FileNotFoundError as e:
            logger.error(f"Error evaluating image: {e}") # More specific error message
            return None
        except Exception as e:
            logger.error(f"Error evaluating image: {str(e)}")
            return None

    async def refine_prompt(self, original_prompt: str, evaluation: Dict) -> Optional[str]:
        """Refine a prompt based on evaluation."""
        try:
            return await self.gemini_client.refine_prompt(original_prompt, evaluation)
        except Exception as e:
            logger.error(f"Error refining prompt: {str(e)}")
            return None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
