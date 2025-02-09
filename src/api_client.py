import aiohttp
import google.generativeai as genai
import logging
import json
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class FalClient:
    """Client for interacting with the fal.ai API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://110602490-fast-sdxl.fal.run"
        self.session: Optional[aiohttp.ClientSession] = None
        self.max_retries = 3
        self.retry_delay = 60  # seconds

    async def setup(self):
        """Initialize the aiohttp session."""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def cleanup(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def generate_image(self, prompt: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Generate an image using fal.ai API.
        
        Args:
            prompt: The text prompt for image generation
            **kwargs: Additional parameters for image generation
            
        Returns:
            Dictionary containing the API response or None if failed
        """
        if not self.session:
            await self.setup()

        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "prompt": prompt,
            **kwargs
        }

        for attempt in range(self.max_retries):
            try:
                async with self.session.post(
                    f"{self.base_url}/generate",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 429:  # Rate limit
                        logger.warning(f"Rate limit hit, waiting {self.retry_delay} seconds...")
                        await asyncio.sleep(self.retry_delay)
                        continue
                        
                    response.raise_for_status()
                    return await response.json()
                    
            except aiohttp.ClientError as e:
                logger.error(f"API request failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
                
        logger.error("Max retries exceeded for image generation")
        return None

class GeminiClient:
    """Client for interacting with Google's Gemini API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.vision_model = genai.GenerativeModel('gemini-pro-vision')
        self.text_model = genai.GenerativeModel('gemini-pro')

    async def evaluate_image(self, image) -> Optional[Dict[str, Any]]:
        """
        Evaluate an image using Gemini Vision API.
        
        Args:
            image: PIL Image object to evaluate
            
        Returns:
            Dictionary containing evaluation results or None if failed
        """
        try:
            evaluation_prompt = """
            Analyze this AI-generated image and provide a detailed evaluation including:
            1. Overall quality and coherence
            2. How well it matches typical expectations
            3. Any notable issues or artifacts
            4. Suggestions for improvement
            
            Format the response as a JSON object with these exact keys:
            {
                "quality": "description of overall quality",
                "coherence": "assessment of coherence",
                "issues": ["list", "of", "issues"],
                "suggestions": ["list", "of", "suggestions"]
            }
            """

            response = await self.vision_model.generate_content([evaluation_prompt, image])
            
            try:
                # Attempt to parse response as JSON
                return json.loads(response.text)
            except json.JSONDecodeError:
                # Fallback structure if response isn't valid JSON
                return {
                    "quality": response.text,
                    "coherence": None,
                    "issues": [],
                    "suggestions": []
                }

        except Exception as e:
            logger.error(f"Error evaluating image with Gemini: {str(e)}")
            return None

    async def refine_prompt(self, original_prompt: str, evaluation: Dict[str, Any]) -> Optional[str]:
        """
        Generate a refined prompt based on evaluation feedback.
        
        Args:
            original_prompt: The original image generation prompt
            evaluation: Dictionary containing image evaluation results
            
        Returns:
            Refined prompt string or None if failed
        """
        try:
            refinement_prompt = f"""
            Original prompt: {original_prompt}
            
            Image evaluation:
            Quality: {evaluation.get('quality', 'N/A')}
            Issues: {', '.join(evaluation.get('issues', []))}
            Suggestions: {', '.join(evaluation.get('suggestions', []))}
            
            Based on this evaluation, generate an improved version of the original prompt
            that addresses the noted issues and incorporates the suggestions.
            Return only the refined prompt text, without any additional explanation.
            """

            response = await self.text_model.generate_content(refinement_prompt)
            return response.text.strip()

        except Exception as e:
            logger.error(f"Error refining prompt with Gemini: {str(e)}")
            return None
