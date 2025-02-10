from typing import Dict, Any
import json
from pathlib import Path
import asyncio
from src.APIClient import GeminiClient
import logging

logger = logging.getLogger(__name__)

class PromptGenerator:
    MODEL_GUIDANCE = {
        "flux": """
        For Flux model, focus on these aspects:
        - Use natural language descriptions that focus on the visual elements
        - Avoid technical terms like "4K", "HDR", etc.
        - Emphasize composition, lighting, and mood naturally
        - Don't use explicit style keywords like "digital art" or "photorealistic"
        - Keep descriptions flowing and conversational
        """,
        "sdxl": """
        For SDXL model, focus on these aspects:
        - Include technical quality terms (8K, HDR, photorealistic)
        - Use explicit artistic style references
        - Include specific camera details (wide angle, macro, etc.)
        - Add detailed lighting descriptions
        - Can use explicit style keywords and artist references
        """
    }

    def __init__(self, gemini_api_key: str):
        self.gemini = GeminiClient(gemini_api_key)
        
    async def generate_prompt(self, title: str, scene: str, mood: str = "", model: str = "flux") -> str:
        """Generate an image prompt using Gemini."""
        try:
            logger.info(f"Generating prompt for title: {title}")
            # Get model-specific guidance
            model_guidance = self.MODEL_GUIDANCE.get(model.lower(), self.MODEL_GUIDANCE["flux"])
            
            system_prompt = f"""You are an expert at writing prompts for AI image generation.
            Your task is to create concise, visually-focused prompts that describe the scene directly.
            
            Guidelines:
            - Focus purely on visual elements (composition, lighting, colors, subjects)
            - Remove any meta-commentary or explanatory text
            - Don't use phrases like "a scene of" or "an image of"
            - Start directly with the subject and action
            - Keep the prompt under 100 words
            - Maintain natural, flowing language
            
            {model_guidance}
            """
            
            user_prompt = f"""
            Convert this scene description into a focused visual prompt.
            
            Title: {title}
            Scene: {scene}
            Mood: {mood}
            
            Respond with ONLY the prompt text, no additional commentary or explanations.
            """
            
            logger.info("Calling Gemini API for prompt generation")
            response = self.gemini.model.generate_content([system_prompt, user_prompt])
            response.resolve()
            
            if not response.text:
                logger.error("No response from Gemini API")
                return ""
                
            logger.info("Successfully generated prompt")
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error generating prompt: {str(e)}", exc_info=True)
            return ""
    
    async def process_json_file(self, input_path: Path, progress_callback=None) -> Dict[str, Any]:
        """Process all scenes in a JSON file and generate prompts."""
        try:
            logger.info(f"Opening input file: {input_path}")
            with open(input_path) as f:
                data = json.load(f)
            logger.info("Successfully loaded JSON data")
            
            if "prompts" not in data:
                logger.error("Invalid JSON format: 'prompts' key not found")
                raise ValueError("Invalid JSON format: 'prompts' key not found")
            
            # Get default model from file or use flux
            default_model = data.get("model", "flux")
            logger.info(f"Using default model: {default_model}")
                
            total_scenes = len(data["prompts"])
            scenes_generated = 0
            scenes_existing = 0
            
            logger.info(f"Processing {total_scenes} scenes")
            for scene in data["prompts"]:
                # Get model for this scene (fallback to default if not specified)
                model = scene.get("model", default_model)
                
                if not scene.get("prompt"):  # Only generate if prompt is empty
                    logger.info(f"Generating prompt for scene {scene.get('id', 'unknown')}")
                    scene["prompt"] = await self.generate_prompt(
                        scene.get("title", ""),
                        scene.get("scene", ""),
                        scene.get("mood", ""),
                        model
                    )
                    # Ensure model is saved in scene data
                    scene["model"] = model
                    scenes_generated += 1
                    if progress_callback:
                        progress_callback(f"Generated prompt {scenes_generated} of {total_scenes} for {model}")
                else:
                    logger.info(f"Prompt already exists for scene {scene.get('id', 'unknown')}")
                    scenes_existing += 1
                    # Ensure model is saved in scene data if missing
                    if "model" not in scene:
                        scene["model"] = default_model
                    
                await asyncio.sleep(1)  # Rate limiting
            
            logger.info("Completed processing all scenes")
            return {
                "data": data,
                "stats": {
                    "total_scenes": total_scenes,
                    "scenes_generated": scenes_generated,
                    "scenes_existing": scenes_existing,
                    "model": default_model
                }
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON file: {str(e)}")
            raise ValueError(f"Invalid JSON file: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing JSON file: {str(e)}", exc_info=True)
            raise Exception(f"Error processing JSON file: {str(e)}")

    async def update_json_file(self, input_path: Path, output_path: Path = None, progress_callback=None) -> Dict[str, Any]:
        """Process JSON file and save results."""
        logger.info(f"Starting JSON file update. Input path: {input_path}")
        if output_path is None:
            output_path = input_path
            
        try:
            result = await self.process_json_file(input_path, progress_callback)
            logger.info(f"Writing updated data to: {output_path}")
            
            with open(output_path, 'w') as f:
                json.dump(result["data"], f, indent=2)
            logger.info("Successfully wrote updated JSON file")
            return result["stats"]
        except Exception as e:
            logger.error(f"Error saving JSON file: {str(e)}", exc_info=True)
            raise Exception(f"Error saving JSON file: {str(e)}")
