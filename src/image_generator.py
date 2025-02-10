import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, Callable
from PIL import Image
import fal_client
from datetime import datetime

from src.image_evaluator import ImageEvaluator
from src.prompt_refiner import PromptRefiner
from src.prompt_handler import PromptHandler

logger = logging.getLogger(__name__)

class ImageGenerator:
    """Handles the complete image generation process."""

    def __init__(self, fal_api_key: str, gemini_api_key: str, output_base_path: Path,
                 batch_size: int = 3, prompt_handler: PromptHandler = None):
        """Initialize the image generator with API clients."""
        self.output_base_path = Path(output_base_path)
        self.batch_size = batch_size
        self.prompt_handler = prompt_handler

        # Initialize API clients
        fal_client.api_key = fal_api_key
        from src.api_client import FalClient
        self.fal_client = FalClient(fal_api_key)
        self.image_evaluator = ImageEvaluator(gemini_api_key)
        self.prompt_refiner = PromptRefiner(gemini_api_key, self.output_base_path.parent, self.prompt_handler) # Pass PromptHandler

    async def setup(self):
        """Initialize resources."""
        await self.fal_client.setup()
        await self.image_evaluator.setup()
        await self.prompt_refiner.setup()
        if self.prompt_refiner:
            await self.prompt_refiner.setup()

    async def cleanup(self):
        """Cleanup resources."""
        await self.fal_client.cleanup()
        await self.image_evaluator.cleanup()
        await self.prompt_refiner.cleanup()

    def _construct_full_prompt(self, prompt_data: Dict) -> str:
        """Construct the full prompt from prompt data."""
        return (
            f"Title: {prompt_data['title']}\n"
            f"Scene: {prompt_data['scene']}\n"
            f"Mood: {prompt_data['mood']}\n"
            f"Prompt: {prompt_data['prompt']}"
        )

    async def process_prompt(self, prompt_id: str, prompt_data: dict,
                           progress_callback: Callable[[str], None]) -> bool:
        """Process a single prompt through the generation pipeline."""
        try:
            def update_progress(message: str):
                progress_callback(message)

            # Generate and evaluate image
            image_path, evaluation = await self.generate_and_evaluate(
                prompt_data,
                prompt_id,
                progress_callback=update_progress
            )

            if not image_path:
                return False

            # Save results - use PromptHandler instance
            if self.prompt_handler:
                await self.prompt_handler.save_results(
                    prompt_id=prompt_id,
                    iteration=2,  # Final iteration
                    image_path=image_path,
                    prompt=prompt_data['prompt'],
                    evaluation=evaluation
                )
            else:
                logger.error("PromptHandler not initialized in ImageGenerator. Cannot save results.")
                return False

            logger.info(f"Successfully completed processing for prompt {prompt_id}")
            return True

        except asyncio.CancelledError:
            logger.warning(f"Processing cancelled for prompt {prompt_id}")
            raise
        except Exception as e:
            logger.error(f"Error processing prompt {prompt_id}: {str(e)}", exc_info=True)
            return False


    async def generate_and_evaluate(self, prompt_data: Dict, prompt_id: str,
                                  progress_callback=None) -> Tuple[Optional[Path], Optional[Dict]]:
        """
        Complete generation process including evaluation and refinement.
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

            refined_prompt = await self.refine_prompt(full_prompt, prompt_id, evaluation) # Pass prompt_id
            if not refined_prompt or "No refinement needed" in refined_prompt:
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
                image_data = result['images'][0]  # Assuming first image
                # Assuming image_data is base64 encoded, decode and save
                binary_data = fal_client.base64.b64decode(image_data)
                with open(image_path, 'wb') as f:
                    f.write(binary_data)
                return image_path

            return None

        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            return None

    async def evaluate_image(self, image_path: Path) -> Optional[Dict]:
        """Evaluate an image using Gemini Vision."""
        try:
            return await self.image_evaluator.evaluate_image(Image.open(image_path))
        except Exception as e:
            logger.error(f"Error evaluating image: {str(e)}")
            return None

    async def refine_prompt(self, original_prompt: str, prompt_id: str, evaluation: Dict) -> Optional[str]: # Added prompt_id
        """Refine a prompt based on evaluation."""
        try:
            return await self.prompt_refiner.refine_prompt(original_prompt, prompt_id, evaluation) # Pass prompt_id
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
