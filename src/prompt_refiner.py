import google.generativeai as genai
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class PromptRefiner:
    """Handles prompt refinement using Google Gemini API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key) # Initialize Gemini API here
        self.model = genai.GenerativeModel('gemini-pro') # Use text model for prompt refinement

    async def setup(self):
        """Initialize the client."""
        pass

    async def cleanup(self):
        """Clean up resources."""
        pass

    async def refine_prompt(self, original_prompt: str, evaluation_result: Dict) -> Optional[str]:
        """Refine a prompt using Google Gemini API based on evaluation."""
        try:
            if not evaluation_result or 'evaluation_text' not in evaluation_result:
                logger.warning("No evaluation result provided for prompt refinement.")
                return original_prompt # Return original prompt if no evaluation

            evaluation_text = evaluation_result['evaluation_text']

            prompt_instruction = (
                "Given the original prompt for generating an image and the evaluation of the generated image, "
                "refine the original prompt to address the issues mentioned in the evaluation and improve the image quality.\n"
                f"Original Prompt: {original_prompt}\n"
                f"Evaluation: {evaluation_text}\n"
                "Refined Prompt:"
            )

            response = self.model.generate_content(prompt_instruction)
            response.resolve()

            if response.text:
                refined_prompt = response.text
                logger.info(f"Prompt refined: {refined_prompt}")
                return refined_prompt
            else:
                logger.warning("No refined prompt received from Gemini API, returning original.")
                return original_prompt # Return original if refinement fails

        except Exception as e:
            logger.error(f"Error refining prompt with Gemini API: {str(e)}")
            return original_prompt # Return original prompt in case of error
