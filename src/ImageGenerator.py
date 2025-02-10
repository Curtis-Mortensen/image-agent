import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, Callable
import fal_client
import base64
from datetime import datetime
import os
import sqlite3
from config import DATABASE_PATH
import aiohttp
from src.PromptHandler import PromptHandler
from src.APIClient import FalClient

logger = logging.getLogger(__name__)

class ImageGenerator:
    """Handles image generation with database tracking."""

    def __init__(self, fal_api_key: str, output_base_path: Path, db_path: str = DATABASE_PATH):
        self.output_base_path = Path(output_base_path)
        self.db_path = db_path
        
        # Initialize FAL client
        fal_client.api_key = fal_api_key
        self.fal_client = FalClient(fal_api_key)

    def _init_db(self):
        """Initialize database table for image tracking."""
        # Database initialization is now handled by DatabaseGenerator
        pass

    async def _save_generation_record(self, prompt_id: str, iteration: int, 
                                    variant: int, image_path: str, prompt_text: str):
        """Save generation record to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO generated_images 
                    (prompt_id, iteration, variant, image_path, prompt_text)
                    VALUES (?, ?, ?, ?, ?)
                """, (prompt_id, iteration, variant, str(image_path), prompt_text))
                logger.debug(f"Saved generation record for {prompt_id}, "
                           f"iteration {iteration}, variant {variant}")
        except Exception as e:
            logger.error(f"Error saving generation record: {str(e)}")
            raise

    async def process_prompt(self, prompt_id: str, prompt_data: dict,
                           iteration: int, 
                           progress_callback: Callable[[str], None]) -> Optional[Path]:
        """Process a single prompt through the generation pipeline."""
        try:
            full_prompt = self._construct_full_prompt(prompt_data)
            progress_callback(f"Generating iteration {iteration} for {prompt_id}")

            image_path = await self.generate_image(
                full_prompt,
                prompt_id,
                iteration=iteration,
                model_id="fal-ai/fast-lightning-sdxl"
            )

            if image_path:
                await self._save_generation_record(
                    prompt_id, iteration, 0, str(image_path), full_prompt
                )
                logger.info(f"Generated image for {prompt_id}, iteration {iteration}")
                return image_path

            return None

        except Exception as e:
            logger.error(f"Error processing {prompt_id}: {str(e)}")
            return None

    async def generate_image(self, prompt: str, prompt_id: str,
                           iteration: int = 1, variant: int = 0, **kwargs) -> Optional[Path]:
        """Generate and save a single image variant."""
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

            try:
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
                            
                        logger.info(f"Saved image to {image_path}")
                        return image_path

            except Exception as e:
                logger.error(f"Error saving image file: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Image generation error: {str(e)}")
            return None

    def _construct_full_prompt(self, prompt_data: Dict) -> str:
        """Construct full prompt from data."""
        return "\n".join([
            f"Title: {prompt_data['title']}",
            f"Scene: {prompt_data['scene']}",
            f"Mood: {prompt_data['mood']}",
            f"Prompt: {prompt_data['prompt']}"
        ])

    async def get_generation_history(self, prompt_id: str) -> list:
        """Retrieve generation history for a prompt."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT iteration, variant, image_path, created_at, status, evaluation_score
                FROM generated_images
                WHERE prompt_id = ?
                ORDER BY iteration, variant
            """, (prompt_id,))
            return cursor.fetchall()

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
