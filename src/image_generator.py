import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image
import base64
import io
import aiofiles
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from src.api_client import FalClient, GeminiClient

logger = logging.getLogger(__name__)

class ImageGenerator:
    """Handles the generation, evaluation, and refinement of images."""
    
    def __init__(self, fal_api_key: str, gemini_api_key: str, output_base_path: Path,
                 image_size: Tuple[int, int] = (1024, 1024), batch_size: int = 3):
        """
        Initialize the image generator.
        
        Args:
            fal_api_key: API key for fal.ai
            gemini_api_key: API key for Gemini
            output_base_path: Base path for outputs
            image_size: Tuple of (width, height) for generated images
            batch_size: Number of concurrent generations
        """
        self.output_base_path = Path(output_base_path)
        self.fal_client = FalClient(fal_api_key)
        self.gemini_client = GeminiClient(gemini_api_key)
        self.image_size = image_size
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=4)  # For CPU-bound tasks

    async def setup(self):
        """Initialize API clients and resources."""
        await self.fal_client.setup()

    async def cleanup(self):
        """Cleanup resources."""
        await self.fal_client.cleanup()
        self.executor.shutdown(wait=True)

    async def _save_image(self, image_data: bytes, prompt_id: str, iteration: int) -> Path:
        """Save generated image using async file operations."""
        output_dir = self.output_base_path / prompt_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_path = output_dir / f"iteration_{iteration}.png"
        
        # Use ThreadPoolExecutor for CPU-bound image processing
        def process_and_save():
            image = Image.open(io.BytesIO(image_data))
            image.save(image_path, format='PNG', optimize=True)
        
        await asyncio.get_event_loop().run_in_executor(
            self.executor, process_and_save
        )
        
        return image_path

    async def _save_metadata(self, prompt_id: str, iteration: int,
                           prompt: str, params: Dict, image_path: Path) -> None:
        """Save metadata using async file operations."""
        metadata = {
            "prompt": prompt,
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "parameters": params,
            "image_path": str(image_path)
        }
        
        metadata_path = image_path.parent / f"iteration_{iteration}_metadata.json"
        async with aiofiles.open(metadata_path, 'w') as f:
            await f.write(json.dumps(metadata, indent=2))

    async def generate_image(self, prompt: str, prompt_id: str,
                           iteration: int = 1, **kwargs) -> Optional[Path]:
        """Generate an image with full parameter control."""
        try:
            params = {
                "prompt": prompt,
                "negative_prompt": kwargs.get('negative_prompt', 
                    "blurry, low quality, distorted, deformed"),
                "num_inference_steps": kwargs.get('num_inference_steps', 30),
                "guidance_scale": kwargs.get('guidance_scale', 7.5),
                "width": self.image_size[0],
                "height": self.image_size[1],
                **kwargs
            }
            
            response = await self.fal_client.generate_image(**params)
            if not response:
                return None
            
            image_data = base64.b64decode(response['images'][0])
            image_path = await self._save_image(image_data, prompt_id, iteration)
            await self._save_metadata(prompt_id, iteration, prompt, params, image_path)
            
            return image_path
            
        except Exception as e:
            logger.error(f"Error generating image for prompt {prompt_id}: {str(e)}")
            return None

    async def evaluate_image(self, image_path: Path) -> Optional[Dict]:
        """Evaluate generated image using Gemini Vision."""
        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found at {image_path}")
            
            # Use ThreadPoolExecutor for image loading
            def load_image():
                return Image.open(image_path)
            
            image = await asyncio.get_event_loop().run_in_executor(
                self.executor, load_image
            )
            
            evaluation = await self.gemini_client.evaluate_image(image)
            if evaluation:
                eval_path = image_path.parent / f"{image_path.stem}_evaluation.json"
                async with aiofiles.open(eval_path, 'w') as f:
                    await f.write(json.dumps(evaluation, indent=2))
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating image {image_path}: {str(e)}")
            return None

    async def refine_prompt(self, original_prompt: str, evaluation: Dict) -> Optional[str]:
        """Generate refined prompt with advanced evaluation."""
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
