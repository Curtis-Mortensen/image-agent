import google.generativeai as genai
import logging
from typing import Optional, Dict
import aiofiles
import json
from pathlib import Path
from datetime import datetime

from src.prompt_handler import PromptHandler # Import PromptHandler

logger = logging.getLogger(__name__)

class PromptRefiner:
    """Handles prompt refinement using Google Gemini API, saving adjusted refined prompts."""

    def __init__(self, api_key: str, data_base_path: Path, prompt_handler: PromptHandler):
        self.api_key = api_key
        self.data_base_path = data_base_path
        self.prompt_handler = prompt_handler
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.adherence_phrase = "Image adheres to the prompt"
        self.refined_prompts_dir = self.data_base_path / "refined_prompts"

    async def setup(self):
        """Initialize and ensure refined prompts directory exists."""
        await aiofiles.os.makedirs(self.refined_prompts_dir, exist_ok=True)

    async def cleanup(self):
        """Clean up resources."""
        pass

    async def _get_next_iteration_number(self) -> int:
        """Determine the next iteration number for refined prompts file."""
        max_iteration = 0
        async for filename in aiofiles.os.listdir(self.refined_prompts_dir):
            if filename.startswith("refined_prompts_") and filename.endswith(".json"):
                try:
                    iteration_num = int(filename[len("refined_prompts_"):-len(".json")])
                    max_iteration = max(max_iteration, iteration_num)
                except ValueError:
                    continue
        return max_iteration + 1

    async def _save_refined_prompt(self, prompt_id: str, original_prompt: str, evaluation_text: str, refined_prompt_content: str):
        """Save the blended refined prompt to a JSON file with adjusted format."""
        iteration_num = await self._get_next_iteration_number()
        new_prompt_id = f"{prompt_id}_{iteration_num}"
        refined_prompt_filename = self.refined_prompts_dir / f"refined_prompts_{iteration_num}.json"

        # Get original prompt data from PromptHandler cache
        original_prompt_data = self.prompt_handler.prompts_cache.get(prompt_id)
        if not original_prompt_data:
            logger.error(f"Original prompt data not found for prompt_id: {prompt_id}")
            return

        blended_prompt = refined_prompt_content # Refined prompt is the blended prompt

        refined_prompt_data = {
            "id": new_prompt_id,
            "iteration": iteration_num,
            "timestamp": datetime.now().isoformat(),
            "title": original_prompt_data.title, # Keep original title
            "scene": original_prompt_data.scene, # Keep original scene
            "mood": original_prompt_data.mood, # Keep original mood
            "prompt": blended_prompt, # Use "prompt" field for refined prompt - IMPORTANT CHANGE
            "evaluation_text": evaluation_text
        }

        async with aiofiles.open(refined_prompt_filename, 'w') as f:
            await f.write(json.dumps(refined_prompt_data, indent=2))
        logger.info(f"Refined prompt saved to: {refined_prompt_filename}")


    async def refine_prompt(self, original_prompt: str, prompt_id: str, evaluation_result: Dict) -> Optional[str]:
        """Refine a prompt using Gemini API, saving adjusted format refined prompt."""
        try:
            if not evaluation_result or 'evaluation_text' not in evaluation_result:
                logger.warning("No evaluation result provided for prompt refinement.")
                return "No evaluation provided, cannot refine."

            evaluation_text = evaluation_result['evaluation_text']
            logger.info(f"Prompt Evaluation Text: {evaluation_text}")

            if self.adherence_phrase.lower() in evaluation_text.lower():
                logger.info(f"Gemini evaluation indicates adherence with phrase: '{self.adherence_phrase}'. No refinement needed.")
                return f"Success: {self.adherence_phrase}. No refinement needed."

            prompt_instruction = (
                "You are evaluating if a generated image adheres to a user's prompt. Adherence is relative, minor variations are acceptable.\n"
                "Focus on identifying significant deviations:\n"
                "- Are there major elements *missing* from the image description that were *explicitly requested* in the original prompt?\n"
                "- Are there prominent elements *present* in the image description that were *not requested* or implied by the original prompt?\n"
                "- Are there *large, significant differences* in key elements between the prompt and the image (e.g., wrong subject, wrong scene)?\n"
                "If none of these significant deviations are clearly present, then respond with ONLY this exact phrase: '{self.adherence_phrase}'.\n"
                "Otherwise, if there are significant deviations, suggest a revised prompt that would better guide the image generation model to produce the desired image.\n"
                f"Original Prompt: {original_prompt}\n"
                f"Image Description: {evaluation_text}\n"
                "Analysis and Refinement Suggestion:"
            )

            response = self.model.generate_content(prompt_instruction)
            response.resolve()

            if response.text:
                refined_prompt_content = response.text
                logger.info(f"Prompt refined: {refined_prompt_content}")
                await self._save_refined_prompt(prompt_id, original_prompt, evaluation_text, refined_prompt_content)
                return refined_prompt_content
            else:
                logger.warning("No refined prompt received from Gemini API, returning original.")
                return original_prompt

        except Exception as e:
            logger.error(f"Error refining prompt with Gemini API: {str(e)}")
            return "Error during prompt refinement."
