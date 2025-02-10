"""
Handles pure image description using Google's Gemini Vision API.
"""

import logging
from typing import Optional, Dict
from PIL import Image
import google.generativeai as genai

logger = logging.getLogger(__name__)

class ImageVision:
    """Handles image description using Gemini Vision API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro-vision')

    async def describe_image(self, image: Image.Image) -> Optional[str]:
        """
        Generate a detailed description of the image.
        
        Args:
            image: PIL Image to describe
            
        Returns:
            str: Detailed description of the image, or None if error
        """
        try:
            prompt = """
            Describe this image in detail, focusing on:
            - Main subjects and their characteristics
            - Composition and layout
            - Colors and lighting
            - Style and artistic elements
            - Notable details or unique features
            
            Provide a clear, objective description without interpretation or judgment.
            """
            
            response = self.model.generate_content([prompt, image])
            response.resolve()
            
            if not response.text:
                logger.error("No response from Gemini Vision API")
                return None
                
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error describing image: {str(e)}")
            return None

    async def setup(self):
        """Initialize resources if needed."""
        pass

    async def cleanup(self):
        """Cleanup resources if needed."""
        pass

    async def __aenter__(self):
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
