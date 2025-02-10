import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, Callable
from PIL import Image
import fal_client
import base64 # Import the base64 module
from datetime import datetime
import os # Import the os module

logger = logging.getLogger(__name__)

class ImageGenerator:
    """Handles image generation, saving images directly to outputs/images."""

    def __init__(self, fal_api_key: str, output_base_path: Path,
                 batch_size: int = 3): # Removed gemini_api_key and prompt_handler
        """Initialize the image generator with API clients."""
        self.output_base_path = Path(output_base_path)
        self.batch_size = batch_size

        # Initialize API clients
        fal_client.api_key = fal_api_key
        from src.api_client import FalClient
        self.fal_client = FalClient(fal_api_key)

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

    async def process_prompt(self, prompt_id: str, prompt_data: dict,
                           iteration: int, progress_callback: Callable[[str], None]) -> Optional[Path]: # Added iteration
        """Process a single prompt through the generation pipeline."""
        try:
            def update_progress(message: str):
                progress_callback(message)

            # Generate image
            image_path = await self.generate_image_iteration( # Renamed and simplified function
                prompt_data,
                prompt_id,
                iteration=iteration,
                progress_callback=update_progress
            )

            if not image_path:
                return None

            logger.info(f"Successfully generated image for prompt {prompt_id}, iteration {iteration}")
            return image_path

        except asyncio.CancelledError:
            logger.warning(f"Processing cancelled for prompt {prompt_id}")
            raise
        except Exception as e:
            logger.error(f"Error processing prompt {prompt_id}: {str(e)}", exc_info=True)
            return None


    async def generate_image_iteration(self, prompt_data: Dict, prompt_id: str, iteration: int, progress_callback=None) -> Optional[Path]: # Renamed and simplified
        """
        Generate a single image iteration and return the image path.
        """
        try:
            # Initial generation
            if progress_callback:
                progress_callback(f"Generating iteration {iteration} image for {prompt_id}")

            full_prompt = self._construct_full_prompt(prompt_data)
            image_path = await self.generate_image(
                full_prompt,
                prompt_id,
                iteration=iteration,
                model_id="fal-ai/fast-lightning-sdxl"
            )
            return image_path

        except Exception as e:
            logger.error(f"Error in generate_image_iteration: {str(e)}")
            return None


    async def generate_image(self, prompt: str, prompt_id: str,
                           iteration: int = 1, **kwargs) -> Optional[Path]:
        """Generate a single image, saving directly to outputs/images."""
        try:
            result = await self.fal_client.generate_image(prompt, **kwargs)
            if not result:
                return None

            # Save the image directly to outputs/images
            output_dir = self.output_base_path / "images" # Updated output path: no prompt_id subfolder
            output_dir.mkdir(parents=True, exist_ok=True)
            image_path = output_dir / f"{prompt_id}_iteration_{iteration}.png" # Include prompt_id in filename

            # Save image data
            if 'images' in result and result['images']:
                image_data = result['images'][0]  # Assuming first image
                # Assuming image_data is a dictionary containing 'content' key with base64 string
                binary_data = base64.b64decode(image_data.get('content', '')) # Access 'content' key, default to empty string if not found
                with open(image_path, 'wb') as f:
                    f.write(binary_data)

                if os.path.exists(image_path): # Check if file exists
                    logger.info(f"Image saved successfully to: {image_path}")
                else:
                    logger.error(f"Error saving image to: {image_path}")
                    return None # Return None if image was not saved

                return image_path

            return None

        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            return None


    async def __aenter__(self):
        """Async context manager entry."""
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
