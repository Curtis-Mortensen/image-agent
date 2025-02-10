from typing import Dict, Any
import json
from pathlib import Path
import asyncio
from src.api_client import GeminiClient

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
        # Get model-specific guidance
        model_guidance = self.MODEL_GUIDANCE.get(model.lower(), self.MODEL_GUIDANCE["flux"])
        
        system_prompt = f"""You are an expert at writing prompts for AI image generation. 
        Create a detailed prompt that will result in a high-quality, artistic image.
        Focus on style, composition, lighting, and mood. 
        The prompt should be specific and descriptive, but concise.
        
        {model_guidance}
        """
        
        user_prompt = f"""
        Title: {title}
        Scene Description: {scene}
        Mood: {mood}
        Target Model: {model}
        
        Please generate a detailed image generation prompt based on this information.
        """
        
        response = await self.gemini.generate_text(system_prompt, user_prompt)
        return response.strip()
    
    async def process_json_file(self, input_path: Path, progress_callback=None) -> Dict[str, Any]:
        """Process all scenes in a JSON file and generate prompts."""
        try:
            with open(input_path) as f:
                data = json.load(f)
            
            if "prompts" not in data:
                raise ValueError("Invalid JSON format: 'prompts' key not found")
            
            # Get default model from file or use flux
            default_model = data.get("model", "flux")
                
            total_scenes = len(data["prompts"])
            scenes_generated = 0
            scenes_existing = 0
            
            for scene in data["prompts"]:
                # Get model for this scene (fallback to default if not specified)
                model = scene.get("model", default_model)
                
                if not scene.get("prompt"):  # Only generate if prompt is empty
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
                    scenes_existing += 1
                    # Ensure model is saved in scene data if missing
                    if "model" not in scene:
                        scene["model"] = default_model
                    
                await asyncio.sleep(1)  # Rate limiting
            
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
            raise ValueError(f"Invalid JSON file: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing JSON file: {str(e)}")

    async def update_json_file(self, input_path: Path, output_path: Path = None, progress_callback=None) -> Dict[str, Any]:
        """Process JSON file and save results."""
        if output_path is None:
            output_path = input_path
            
        result = await self.process_json_file(input_path, progress_callback)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(result["data"], f, indent=2)
            return result["stats"]
        except Exception as e:
            raise Exception(f"Error saving JSON file: {str(e)}")
