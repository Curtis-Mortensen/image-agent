import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class PromptHandler:
    """Handles loading prompts and saving generation results."""

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
        self.results_path.mkdir(parents=True, exist_ok=True)

    def _validate_prompt(self, prompt_id: str, prompt_data: Dict) -> bool:
        """
        Validate prompt data structure.
        
        Args:
            prompt_id: Unique identifier for the prompt
            prompt_data: Dictionary containing prompt data
            
        Returns:
            bool indicating if prompt is valid
        """
        required_fields = ['title', 'scene', 'mood', 'prompt']
        
        for field in required_fields:
            if field not in prompt_data:
                logger.error(f"Prompt {prompt_id} missing required field: {field}")
                return False
            
            if not isinstance(prompt_data[field], str):
                logger.error(f"Prompt {prompt_id} field {field} must be string")
                return False
            
            if not prompt_data[field].strip():
                logger.error(f"Prompt {prompt_id} field {field} cannot be empty")
                return False
                
        return True

    def load_prompts(self) -> Dict[str, Dict[str, str]]:
        """
        Load and validate prompts from input JSON file.
        
        Returns:
            Dictionary mapping prompt IDs to prompt data
        """
        try:
            if not self.input_file_path.exists():
                raise FileNotFoundError(f"Input file not found: {self.input_file_path}")

            with open(self.input_file_path, 'r') as f:
                data = json.load(f)

            if not isinstance(data, dict) or 'prompts' not in data:
                raise ValueError("Input JSON must contain 'prompts' key")

            prompts = {}
            for idx, prompt_data in enumerate(data['prompts'], 1):
                # Generate ID if not provided
                prompt_id = str(prompt_data.get('id', f'prompt_{idx:03d}'))
                
                if self._validate_prompt(prompt_id, prompt_data):
                    prompts[prompt_id] = prompt_data
                else:
                    logger.warning(f"Skipping invalid prompt: {prompt_id}")

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

    def _create_prompt_directory(self, prompt_id: str) -> Path:
        """
        Create and return directory for prompt outputs.
        
        Args:
            prompt_id: Unique identifier for the prompt
            
        Returns:
            Path to prompt directory
        """
        prompt_dir = self.results_path / prompt_id
        prompt_dir.mkdir(parents=True, exist_ok=True)
        return prompt_dir

    def save_results(self,
                    prompt_id: str,
                    iteration: int,
                    image_path: Optional[Path],
                    prompt: str,
                    evaluation: Optional[Dict[str, Any]] = None) -> None:
        """
        Save generation results for a prompt iteration.
        
        Args:
            prompt_id: Unique identifier for the prompt
            iteration: Iteration number
            image_path: Path to generated image
            prompt: The prompt used for generation
            evaluation: Optional evaluation results
        """
        try:
            prompt_dir = self._create_prompt_directory(prompt_id)
            
            # Prepare results data
            results = {
                "timestamp": datetime.now().isoformat(),
                "iteration": iteration,
                "prompt": prompt,
                "image_path": str(image_path) if image_path else None,
                "evaluation": evaluation
            }

            # Save results JSON
            results_file = prompt_dir / f"iteration_{iteration}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            # Create iteration summary
            self._update_summary(prompt_id, iteration, results)

            logger.debug(f"Saved results for prompt {prompt_id}, iteration {iteration}")

        except Exception as e:
            logger.error(f"Error saving results for prompt {prompt_id}: {str(e)}")

    def _update_summary(self, prompt_id: str, iteration: int, results: Dict) -> None:
        """
        Update the summary file for a prompt.
        
        Args:
            prompt_id: Unique identifier for the prompt
            iteration: Iteration number
            results: Results data to add to summary
        """
        try:
            prompt_dir = self._create_prompt_directory(prompt_id)
            summary_file = prompt_dir / "summary.json"
            
            # Load existing summary or create new
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
            else:
                summary = {
                    "prompt_id": prompt_id,
                    "iterations": {},
                    "last_updated": None
                }

            # Update summary with new iteration
            summary["iterations"][str(iteration)] = {
                "timestamp": results["timestamp"],
                "image_path": results["image_path"],
                "has_evaluation": bool(results.get("evaluation"))
            }
            summary["last_updated"] = datetime.now().isoformat()

            # Save updated summary
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

        except Exception as e:
            logger.error(f"Error updating summary for prompt {prompt_id}: {str(e)}")

    def get_generation_status(self, prompt_id: str) -> Dict[str, Any]:
        """
        Get the current generation status for a prompt.
        
        Args:
            prompt_id: Unique identifier for the prompt
            
        Returns:
            Dictionary containing generation status
        """
        try:
            prompt_dir = self.results_path / prompt_id
            summary_file = prompt_dir / "summary.json"
            
            if not summary_file.exists():
                return {
                    "prompt_id": prompt_id,
                    "status": "not_started",
                    "iterations_completed": 0
                }

            with open(summary_file, 'r') as f:
                summary = json.load(f)

            return {
                "prompt_id": prompt_id,
                "status": "completed" if len(summary["iterations"]) >= 2 else "in_progress",
                "iterations_completed": len(summary["iterations"]),
                "last_updated": summary["last_updated"]
            }

        except Exception as e:
            logger.error(f"Error getting status for prompt {prompt_id}: {str(e)}")
            return {
                "prompt_id": prompt_id,
                "status": "error",
                "error": str(e)
            }
