import google.generativeai as genai
import logging
from typing import Optional, Dict, Any
from PIL import Image

logger = logging.getLogger(__name__)

class ImageEvaluator:
    """Handles image evaluation using Google Gemini API, using Gemini Flash 2.0 model."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key) # Initialize Gemini API here
        self.model = genai.GenerativeModel('gemini-2.0-flash') # Use Gemini Flash 2.0 model

    async def setup(self):
        """Initialize the client."""
        pass

    async def cleanup(self):
        """Clean up resources."""
        pass

    async def evaluate_image(self, image: Image.Image) -> Optional[Dict]:
        """Evaluate an image using Google Gemini API by providing a detailed description with Gemini Flash 2.0."""
        try:
            if not isinstance(image, Image.Image):
                raise ValueError("Input must be a PIL Image object.")

            response = self.model.generate_content([
                "Describe this image in detail. Focus on the visual elements present, "
                "such as objects, colors, lighting, composition, and overall scene. "
                "The goal is to create a detailed textual representation of the image content." ,
                image
            ])
            response.resolve()

            if response.parts:
                evaluation_text = response.text
                logger.info(f"Image description (Gemini Flash 2.0): {evaluation_text}")
                return {"evaluation_text": evaluation_text} # Structure the output
            else:
                logger.warning("No description text received from Gemini API (Gemini Flash 2.0).")
                return None

        except Exception as e:
            logger.error(f"Error describing image with Gemini API (Gemini Flash 2.0): {str(e)}")
            return None
