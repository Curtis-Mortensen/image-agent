import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, Callable, List
import fal_client
import aiohttp
from config import DATABASE_PATH, PIPELINE_CONFIG
from src.APIClient import FalClient

logger = logging.getLogger(__name__)

class BatchGenerator:
    """Handles batch image generation with multiple variants per prompt."""

    def __init__(self, fal_api_key: str, output_base_path: Path, db_path: str = DATABASE_PATH):
        self.output_base_path = Path(output_base_path)
        self.db_path = db_path
        self.batch_config = PIPELINE_CONFIG["batch_size"]
        
        # Initialize FAL client
        fal_client.api_key = fal_api_key
        self.fal_client = FalClient(fal_api_key)

    def _get_batch_size(self, requested_size: Optional[int] = None) -> int:
        """
        Determine batch size within configured limits.
        
        Args:
            requested_size: Optional specific batch size request
            
        Returns:
            int: Batch size to use, within configured min/max
        """
        size = requested_size or self.batch_config["default"]
        return max(self.batch_config["min"], 
                  min(size, self.batch_config["max"]))

    async def generate_batch(self, prompt_id: str, prompt_data: dict,
                           iteration: int, batch_size: Optional[int] = None,
                           progress_callback: Optional[Callable[[str], None]] = None) -> List[Dict]:
        """
        Generate multiple variants for a single prompt.
        
        Args:
            prompt_id: Unique identifier for the prompt
            prompt_data: Dictionary containing prompt details
            iteration: Current iteration number
            batch_size: Optional specific batch size to use
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of dictionaries containing generated image details
        """
        results = []
        full_prompt = self._construct_full_prompt(prompt_data)
        size = self._get_batch_size(batch_size)
        
        for variant in range(size):
            if progress_callback:
                progress_callback(f"Generating variant {variant + 1}/{size} for iteration {iteration}")
            
            try:
                image_path = await self._generate_variant(
                    full_prompt,
                    prompt_id,
                    iteration=iteration,
                    variant=variant
                )
                
                if image_path:
                    results.append({
                        'prompt_id': prompt_id,
                        'iteration': iteration,
                        'variant': variant,
                        'image_path': str(image_path),
                        'prompt_text': full_prompt
                    })
                    logger.info(f"Generated variant {variant} for {prompt_id}, iteration {iteration}")
            
            except Exception as e:
                logger.error(f"Error generating variant {variant} for {prompt_id}: {str(e)}")
                continue
        
        return results

    async def _generate_variant(self, prompt: str, prompt_id: str,
                              iteration: int = 1, variant: int = 0, **kwargs) -> Optional[Path]:
        """Generate and save a single variant."""
        try:
            result = await self.fal_client.generate_image(prompt, **kwargs)
            if not result or 'images' not in result or not result['images']:
                logger.error("Invalid or empty response from API")
                return None

            image_data = result['images'][0]
            if not image_data or 'url' not in image_data:
                logger.error("Invalid image data in response")
                return None

            output_dir = self.output_base_path / "images"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # New naming convention: {prompt_id}_iter{iteration}_v{variant}.png
            image_path = output_dir / f"{prompt_id}_iter{iteration}_v{variant}.png"

            async with aiohttp.ClientSession() as session:
                async with session.get(image_data['url']) as response:
                    if response.status != 200:
                        logger.error(f"Failed to download image: {response.status}")
                        return None
                        
                    image_content = await response.read()
                    if not image_content:
                        logger.error("Empty image content")
                        return None
                        
                    with open(image_path, 'wb') as f:
                        f.write(image_content)
                        f.flush()
                        
                    if not image_path.exists() or image_path.stat().st_size == 0:
                        logger.error(f"Image file not created or empty")
                        return None
                        
                    logger.info(f"Saved variant to {image_path}")
                    return image_path

        except Exception as e:
            logger.error(f"Error generating variant: {str(e)}")
            return None

    def _construct_full_prompt(self, prompt_data: Dict) -> str:
        """Construct full prompt from data."""
        return "\n".join([
            f"Title: {prompt_data['title']}",
            f"Scene: {prompt_data['scene']}",
            f"Mood: {prompt_data['mood']}",
            f"Prompt: {prompt_data['prompt']}"
        ])

    async def setup(self):
        """Initialize resources."""
        await self.fal_client.setup()

    async def cleanup(self):
        """Cleanup resources."""
        await self.fal_client.cleanup()

    async def __aenter__(self):
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup() 