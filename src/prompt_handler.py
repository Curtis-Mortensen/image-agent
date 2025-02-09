import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import aiofiles
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiofiles.os
import yaml  # for flexible config handling
from dataclasses import dataclass, asdict
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)

@dataclass
class PromptData:
    """Data class for structured prompt data."""
    id: str
    title: str
    scene: str
    mood: str
    prompt: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'PromptData':
        return cls(**data)

class PromptHandler:
    """Handles loading prompts and saving generation results with async support."""

    # JSON schema for prompt validation
    PROMPT_SCHEMA = {
        "type": "object",
        "properties": {
            "prompts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "scene": {"type": "string"},
                        "mood": {"type": "string"},
                        "prompt": {"type": "string"}
                    },
                    "required": ["title", "scene", "mood", "prompt"]
                }
            }
        },
        "required": ["prompts"]
    }

    def __init__(self, input_file_path: Path, output_base_path: Path):
        """
        Initialize the prompt handler.
        
        Args:
            input_file_path: Path to JSON file containing prompts
            output_base_path: Base path for saving outputs
        """
        self.input_file_path = Path(input_file_path)
        self.output_base_path = Path(output_base_path)
        self.results_path = self.output_base_path / "results"
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.prompts_cache: Dict[str, PromptData] = {}

    async def setup(self):
        """Initialize resources and create necessary directories."""
        await aiofiles.os.makedirs(self.results_path, exist_ok=True)

    async def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)

    def _validate_prompt(self, prompt_id: str, prompt_data: Dict) -> bool:
        """
        Validate prompt data structure.
        
        Args:
            prompt_id: Unique identifier for the prompt
            prompt_data: Dictionary containing prompt data
            
        Returns:
            bool indicating if prompt is valid
        """
        try:
            # Convert to PromptData for validation
            PromptData.from_dict(prompt_data)
            return True
        except (ValidationError, KeyError) as e:
            logger.error(f"Invalid prompt data for {prompt_id}: {str(e)}")
            return False

    async def load_prompts(self) -> Dict[str, Dict[str, str]]:
        """
        Load and validate prompts from input JSON file asynchronously.
        
        Returns:
            Dictionary mapping prompt IDs to prompt data
        """
        try:
            if not await aiofiles.os.path.exists(self.input_file_path):
                raise FileNotFoundError(f"Input file not found: {self.input_file_path}")

            async with aiofiles.open(self.input_file_path, 'r') as f:
                content = await f.read()
                data = json.loads(content)

            try:
                validate(instance=data, schema=self.PROMPT_SCHEMA)
            except ValidationError as e:
                logger.error(f"Invalid prompt file structure: {str(e)}")
                return {}

            prompts = {}
            for idx, prompt_data in enumerate(data['prompts'], 1):
                prompt_id = str(prompt_data.get('id', f'prompt_{idx:03d}'))
                
                if self._validate_prompt(prompt_id, prompt_data):
                    prompts[prompt_id] = prompt_data
                    # Cache the validated prompt data
                    self.prompts_cache[prompt_id] = PromptData.from_dict(prompt_data)

            if not prompts:
                logger.error("No valid prompts found in input file")
                return {}

            logger.info(f"Successfully loaded {len(prompts)} prompts")
            return prompts

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing input JSON: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            return {}

    async def _create_prompt_directory(self, prompt_id: str) -> Path:
        """Create and return directory for prompt outputs asynchronously."""
        prompt_dir = self.results_path / prompt_id
        await aiofiles.os.makedirs(prompt_dir, exist_ok=True)
        return prompt_dir

    async def save_results(self,
                          prompt_id: str,
                          iteration: int,
                          image_path: Optional[Path],
                          prompt: str,
                          evaluation: Optional[Dict[str, Any]] = None) -> None:
        """
        Save generation results for a prompt iteration asynchronously.
        
        Args:
            prompt_id: Unique identifier for the prompt
            iteration: Iteration number
            image_path: Path to generated image
            prompt: The prompt used for generation
            evaluation: Optional evaluation results
        """
        try:
            prompt_dir = await self._create_prompt_directory(prompt_id)
            
            # Prepare results data
            results = {
                "timestamp": datetime.now().isoformat(),
                "iteration": iteration,
                "prompt": prompt,
                "image_path": str(image_path) if image_path else None,
                "evaluation": evaluation,
                "prompt_data": asdict(self.prompts_cache.get(prompt_id)) if prompt_id in self.prompts_cache else None
            }

            # Save results as both JSON and YAML for flexibility
            results_file_json = prompt_dir / f"iteration_{iteration}_results.json"
            results_file_yaml = prompt_dir / f"iteration_{iteration}_results.yaml"

            async with aiofiles.open(results_file_json, 'w') as f:
                await f.write(json.dumps(results, indent=2))

            # Save YAML version asynchronously
            def save_yaml():
                with open(results_file_yaml, 'w') as f:
                    yaml.dump(results, f, default_flow_style=False)

            await asyncio.get_event_loop().run_in_executor(
                self.executor, save_yaml
            )

            # Update summary
            await self._update_summary(prompt_id, iteration, results)

            logger.debug(f"Saved results for prompt {prompt_id}, iteration {iteration}")

        except Exception as e:
            logger.error(f"Error saving results for prompt {prompt_id}: {str(e)}")

    async def _update_summary(self, prompt_id: str, iteration: int, results: Dict) -> None:
        """Update the summary file for a prompt asynchronously."""
        try:
            prompt_dir = await self._create_prompt_directory(prompt_id)
            summary_file = prompt_dir / "summary.json"
            
            # Load existing summary or create new
            summary = {"prompt_id": prompt_id, "iterations": {}, "last_updated": None}
            if await aiofiles.os.path.exists(summary_file):
                async with aiofiles.open(summary_file, 'r') as f:
                    content = await f.read()
                    summary = json.loads(content)

            # Update summary with new iteration
            summary["iterations"][str(iteration)] = {
                "timestamp": results["timestamp"],
                "image_path": results["image_path"],
                "has_evaluation": bool(results.get("evaluation")),
                "status": "completed" if results.get("evaluation") else "generated"
            }
            summary["last_updated"] = datetime.now().isoformat()
            summary["total_iterations"] = len(summary["iterations"])

            # Save updated summary
            async with aiofiles.open(summary_file, 'w') as f:
                await f.write(json.dumps(summary, indent=2))

        except Exception as e:
            logger.error(f"Error updating summary for prompt {prompt_id}: {str(e)}")

    async def get_generation_status(self, prompt_id: str) -> Dict[str, Any]:
        """Get the current generation status for a prompt asynchronously."""
        try:
            prompt_dir = self.results_path / prompt_id
            summary_file = prompt_dir / "summary.json"
            
            if not await aiofiles.os.path.exists(summary_file):
                return {
                    "prompt_id": prompt_id,
                    "status": "not_started",
                    "iterations_completed": 0,
                    "prompt_data": asdict(self.prompts_cache.get(prompt_id)) if prompt_id in self.prompts_cache else None
                }

            async with aiofiles.open(summary_file, 'r') as f:
                content = await f.read()
                summary = json.loads(content)

            return {
                "prompt_id": prompt_id,
                "status": "completed" if len(summary["iterations"]) >= 2 else "in_progress",
                "iterations_completed": len(summary["iterations"]),
                "last_updated": summary["last_updated"],
                "prompt_data": asdict(self.prompts_cache.get(prompt_id)) if prompt_id in self.prompts_cache else None
            }

        except Exception as e:
            logger.error(f"Error getting status for prompt {prompt_id}: {str(e)}")
            return {
                "prompt_id": prompt_id,
                "status": "error",
                "error": str(e)
            }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
