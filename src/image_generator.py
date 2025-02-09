import os
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import aiohttp
import google.generativeai as genai
from PIL import Image
import base64
import io

logger = logging.getLogger(__name__)

class ImageGenerator:
    # Default image size
    IMAGE_SIZE = (1024, 1024)
    BATCH_SIZE = 3  # Default batch size for concurrent processing

    def __init__(self, fal_api_key: str, gemini_api_key: str, output_base_path: Path):
        """Initialize the image generator with API keys and output path."""
        self.fal_api_key = fal_api_key
        self.output_base_path = output_base_path
        self.session = None  # Will be initialized in setup
        
        # Initialize Gemini
        genai.configure(api_key=gemini_api_key)
        self.vision_model = genai.GenerativeModel('gemini-pro-vision')
        self.text_model = genai.GenerativeModel('gemini-pro')
        
        # FAL.AI API endpoints
        self.FAL_API_URL = "https://110602490-fast-sdxl.fal.run/generate"

    async def setup(self):
        """Initialize async resources."""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def cleanup(self):
        """Cleanup async resources."""
        if self.session:
            await self.session.close()
            self.session = None

    async def generate_image(self, prompt: str, prompt_id: str, iteration: int = 1) -> Optional[Path]:
        """Generate an image using FAL.AI API and save it."""
        try:
            if not self.session:
                await self.setup()

            headers = {
                "Authorization": f"Key {self.fal_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "prompt": prompt,
                "negative_prompt": "blurry, low quality, distorted, deformed",
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "width": self.IMAGE_SIZE[0],
                "height": self.IMAGE_SIZE[1]
            }
            
            async with self.session.post(self.FAL_API_URL, headers=headers, json=payload) as response:
                if response.status == 429:  # Rate limit
                    await self._handle_rate_limit()
                    return await self.generate_image(prompt, prompt_id, iteration)
                
                response.raise_for_status()
                response_data = await response.json()
            
            # Decode base64 image
            image_data = base64.b64decode(response_data['images'][0])
            image = Image.open(io.BytesIO(image_data))
            
            # Save image
            output_dir = self.output_base_path / prompt_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            image_path = output_dir / f"iteration_{iteration}.png"
            image.save(image_path)
            
            # Save metadata
            metadata = {
                "prompt": prompt,
                "iteration": iteration,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "parameters": payload
            }
            
            metadata_path = output_dir / f"iteration_{iteration}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return image_path
            
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return None
            
    async def evaluate_image(self, image_path: Path) -> Optional[Dict]:
        """Evaluate generated image using Gemini Vision API."""
        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found at {image_path}")
                
            image = Image.open(image_path)
            
            evaluation_prompt = """
            Analyze this AI-generated image and provide a detailed evaluation including:
            1. Overall quality and coherence
            2. How well it matches typical expectations
            3. Any notable issues or artifacts
            4. Suggestions for improvement
            Provide the response as a JSON with these keys: quality, coherence, issues, suggestions
            """
            
            response = await self.vision_model.generate_content([evaluation_prompt, image])
            
            try:
                evaluation = json.loads(response.text)
            except json.JSONDecodeError:
                # Fallback if response isn't valid JSON
                evaluation = {
                    "quality": response.text,
                    "coherence": None,
                    "issues": None,
                    "suggestions": None
                }
            
            # Save evaluation
            output_dir = image_path.parent
            eval_path = output_dir / f"{image_path.stem}_evaluation.json"
            with open(eval_path, 'w') as f:
                json.dump(evaluation, f, indent=2)
                
            return evaluation
                
        except Exception as e:
            print(f"Error evaluating image: {str(e)}")
            return None
            
    async def refine_prompt(self, original_prompt: str, evaluation: Dict) -> Optional[str]:
        """Generate a refined prompt based on the evaluation."""
        try:
            refinement_prompt = f"""
            Original prompt: {original_prompt}
            
            Image evaluation:
            Quality: {evaluation.get('quality', 'N/A')}
            Issues: {evaluation.get('issues', 'N/A')}
            Suggestions: {evaluation.get('suggestions', 'N/A')}
            
            Based on this evaluation, please generate an improved version of the original prompt
            that addresses the noted issues and incorporates the suggestions.
            Provide only the refined prompt text without any explanation.
            """
            
            response = await self.text_model.generate_content(refinement_prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"Error refining prompt: {str(e)}")
            return None
            
    async def _handle_rate_limit(self, retry_after: int = 60):
        """Handle rate limiting by waiting."""
        print(f"Rate limit reached. Waiting {retry_after} seconds...")
        await asyncio.sleep(retry_after)logger = logging.getLogger(__name__)
