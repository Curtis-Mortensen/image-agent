import logging
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image
import base64
import io

from src.api_client import FalClient, GeminiClient

logger = logging.getLogger(__name__)

class ImageGenerator:
    """Handles the generation, evaluation, and refinement of images."""
    
    IMAGE_SIZE = (1024, 1024)
    BATCH_SIZE = 3
    
    def __init__(self, fal_api_key: str, gemini_api_key: str, output_base_path: Path):
        """
        Initialize the image generator with API clients and configuration.
        
        Args:
            fal_api_key: API key for fal.ai
            gemini_api_key: API key for Google's Gemini
            output_base_path: Base path for saving outputs
        """
        self.output_base_path = Path(output_base_path)
        self.fal_client = FalClient(fal_api_key)
        self.gemini_client = GeminiClient(gemini_api_key)
        
        # Default generation parameters
        self.default_params = {
            "negative_prompt": "blurry, low quality, distorted, deformed, ugly, bad anatomy",
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "width": self.IMAGE_SIZE[0],
            "height": self.IMAGE_SIZE[1]
        }

    async def setup(self):
        """Initialize API clients."""
        await self.fal_client.setup()

    async def cleanup(self):
        """Cleanup API clients."""
        await self.fal_client.cleanup()

    def _save_image(self, image_data: bytes, prompt_id: str, iteration: int) -> Path:
        """
        Save generated image and its metadata.
        
        Args:
            image_data: Raw image data in bytes
            prompt_id: Unique identifier for the prompt
            iteration: Iteration number
            
        Returns:
            Path to saved image
        """
        output_dir = self.output_base_path / prompt_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image
        image = Image.open(io.BytesIO(image_data))
        image_path = output_dir / f"iteration_{iteration}.png"
        image.save(image_path)
        
        return image_path

    def _save_metadata(self, 
                      prompt_id: str, 
                      iteration: int, 
                      prompt: str, 
                      params: Dict,
                      image_path: Path) -> None:
        """
        Save metadata for generated image.
        
        Args:
            prompt_id: Unique identifier for the prompt
            iteration: Iteration number
            prompt: The prompt used for generation
            params: Generation parameters used
            image_path: Path to the generated image
        """
        metadata = {
            "prompt": prompt,
            "iteration": iteration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": params,
            "image_path": str(image_path)
        }
        
        metadata_path = image_path.parent / f"iteration_{iteration}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    async def generate_image(self, 
                           prompt: str, 
                           prompt_id: str, 
                           iteration: int = 1,
                           **kwargs) -> Optional[Path]:
        """
        Generate an image using the configured API.
        
        Args:
            prompt: Text prompt for image generation
            prompt_id: Unique identifier for the prompt
            iteration: Iteration number
            **kwargs: Additional generation parameters
            
        Returns:
            Path to generated image or None if failed
        """
        try:
            # Merge default params with any provided kwargs
            params = {**self.default_params, **kwargs, "prompt": prompt}
            
            # Generate image
            response = await self.fal_client.generate_image(**params)
            if not response:
                logger.error(f"Failed to generate image for prompt {prompt_id}")
                return None
            
            # Decode and save image
            image_data = base64.b64decode(response['images'][0])
            image_path = self._save_image(image_data, prompt_id, iteration)
            
            # Save metadata
            self._save_metadata(prompt_id, iteration, prompt, params, image_path)
            
            return image_path
            
        except Exception as e:
            logger.error(f"Error generating image for prompt {prompt_id}: {str(e)}")
            return None

    async def evaluate_image(self, image_path: Path) -> Optional[Dict]:
        """
        Evaluate generated image using Gemini Vision.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing evaluation results or None if failed
        """
        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found at {image_path}")
            
            image = Image.open(image_path)
            evaluation = await self.gemini_client.evaluate_image(image)
            
            if evaluation:
                # Save evaluation results
                eval_path = image_path.parent / f"{image_path.stem}_evaluation.json"
                with open(eval_path, 'w') as f:
                    json.dump(evaluation, f, indent=2)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating image {image_path}: {str(e)}")
            return None

    async def refine_prompt(self, original_prompt: str, evaluation: Dict) -> Optional[str]:
        """
        Generate refined prompt based on evaluation.
        
        Args:
            original_prompt: Original generation prompt
            evaluation: Dictionary containing image evaluation
            
        Returns:
            Refined prompt string or None if failed
        """
        try:
            refined_prompt = await self.gemini_client.refine_prompt(original_prompt, evaluation)
            return refined_prompt
            
        except Exception as e:
            logger.error(f"Error refining prompt: {str(e)}")
            return None
