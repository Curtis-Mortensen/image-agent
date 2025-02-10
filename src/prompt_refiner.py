import google.generativeai as genai
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class PromptRefiner:
    """Handles prompt refinement using Google Gemini API, using Gemini Flash 2.0 model and relative adherence check."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key) # Initialize Gemini API here
        self.model = genai.GenerativeModel('gemini-2.0-flash') # Use Gemini Flash 2.0 model
        self.adherence_phrase = "Image adheres to the prompt" # Define the specific adherence phrase

    async def setup(self):
        """Initialize the client."""
        pass

    async def cleanup(self):
        """Clean up resources."""
        pass

    async def refine_prompt(self, original_prompt: str, evaluation_result: Dict) -> Optional[str]:
        """Refine a prompt using Google Gemini API based on evaluation, checking for relative adherence."""
        try:
            if not evaluation_result or 'evaluation_text' not in evaluation_result:
                logger.warning("No evaluation result provided for prompt refinement.")
                return "No evaluation provided, cannot refine." # Indicate no refinement

            evaluation_text = evaluation_result['evaluation_text']
            logger.info(f"Prompt Evaluation Text: {evaluation_text}")

            # --- Phrase-Based Adherence Check Logic ---
            if self.adherence_phrase.lower() in evaluation_text.lower():
                logger.info(f"Gemini evaluation indicates adherence with phrase: '{self.adherence_phrase}'. No refinement needed.")
                return f"Success: {self.adherence_phrase}. No refinement needed." # Success message with phrase

            # --- Refinement Logic (if not adhering - phrase not found) ---
            prompt_instruction = (
                "You are evaluating if a generated image adheres to a user's prompt. Adherence is relative, minor variations are acceptable.\n"
                "For example, if the prompt was 'a cat riding a horse' and the image is 'an orange cat riding a horse', that is considered adhering.\n"
                "Focus on identifying significant deviations:\n"
                "- Are there major elements *missing* from the image description that were *explicitly requested* in the original prompt?\n"
                "- Are there prominent elements *present* in the image description that were *not requested* or implied by the original prompt?\n"
                "- Are there *large, significant differences* in key elements between the prompt and the image (e.g., wrong subject, wrong scene)?\n"
                "If none of these significant deviations are clearly present in the image description, then respond with ONLY this exact phrase: '{self.adherence_phrase}'.\n"
                "Otherwise, if there are significant deviations, suggest a revised prompt that would better guide the image generation model to produce the desired image.\n"
                f"Original Prompt: {original_prompt}\n"
                f"Image Description: {evaluation_text}\n"
                "Analysis and Refinement Suggestion:"
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
            return "Error during prompt refinement." # Indicate error
