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

logger = logging.getLogger(__name__)

class FalClient:
    """Client for interacting with the fal.ai API."""
    
    def __init__(self, api_key: str, timeout: int = 60):
        self.api_key = api_key
        self.base_url = "https://110602490-fast-sdxl.fal.run"
        self.timeout = ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None

    async def setup(self):
        """Initialize the aiohttp session with connection pooling."""
        if not self.session:
            connector = aiohttp.TCPConnector(limit=10, force_close=False)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.timeout,
                headers={"Authorization": f"Key {self.api_key}"}
            )

    async def cleanup(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3
    )
    async def generate_image(self, prompt: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Generate an image using fal.ai API with automatic retries.
        
        Args:
            prompt: The text prompt for image generation
            **kwargs: Additional parameters for image generation
            
        Returns:
            Dictionary containing the API response or None if failed
        """
        if not self.session:
            await self.setup()

        payload = {
            "prompt": prompt,
            **kwargs
        }

        try:
            async with self.session.post(
                f"{self.base_url}/generate",
                json=payload,
                raise_for_status=True
            ) as response:
                return await response.json()

        except aiohttp.ClientResponseError as e:
            if e.status == 429:  # Rate limit
                retry_after = int(e.headers.get('Retry-After', 60))
                logger.warning(f"Rate limit hit, waiting {retry_after} seconds...")
                await asyncio.sleep(retry_after)
                return await self.generate_image(prompt, **kwargs)
            raise

class GeminiClient:
    """Client for interacting with Google's Gemini API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.vision_model = genai.GenerativeModel('gemini-pro-vision')
        self.text_model = genai.GenerativeModel('gemini-pro')
        
        # Configure safety settings
        self.safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def evaluate_image(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        """
        Evaluate an image using Gemini Vision API with automatic retries.
        
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

            response = await self.vision_model.generate_content(
                [evaluation_prompt, image],
                safety_settings=self.safety_settings
            )
            
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                return {
                    "quality": response.text,
                    "coherence": None,
                    "issues": [],
                    "suggestions": []
                }

        except Exception as e:
            logger.error(f"Error evaluating image with Gemini: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def refine_prompt(self, original_prompt: str, evaluation: Dict[str, Any]) -> Optional[str]:
        """
        Generate a refined prompt based on evaluation feedback with automatic retries.
        
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

            response = await self.text_model.generate_content(
                refinement_prompt,
                safety_settings=self.safety_settings
            )
            return response.text.strip()

        except Exception as e:
            logger.error(f"Error refining prompt with Gemini: {str(e)}")
            raise

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass  # Cleanup if needed
