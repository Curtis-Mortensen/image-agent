"""
Handles grading and scoring of image evaluations based on descriptions and prompts.
"""

import logging
from typing import Dict, Optional
from config import PIPELINE_CONFIG

logger = logging.getLogger(__name__)

class EvaluationGrader:
    """Handles evaluation scoring and pass/fail determination."""

    def __init__(self):
        self.quality_threshold = PIPELINE_CONFIG["quality_threshold"]
        self.refinement_threshold = PIPELINE_CONFIG["refinement_threshold"]

    async def grade_evaluation(self, description: str, original_prompt: str) -> Dict:
        """
        Grade an image evaluation based on its description and original prompt.
        
        Args:
            description: The image description from ImageVision
            original_prompt: The original prompt used to generate the image
            
        Returns:
            Dict containing:
                - score (float): 0-1 score indicating quality
                - passed (bool): Whether image meets minimum quality threshold
                - needs_refinement (bool): Whether prompt needs refinement
                - feedback (str): Specific feedback about the evaluation
        """
        try:
            # Calculate basic matching score based on key elements present
            prompt_elements = set(original_prompt.lower().split())
            description_elements = set(description.lower().split())
            
            # Basic scoring based on overlap of key terms
            common_elements = prompt_elements.intersection(description_elements)
            score = len(common_elements) / len(prompt_elements)
            
            # Adjust score based on description length and completeness
            description_length_factor = min(len(description.split()) / 50, 1.0)
            score = score * 0.7 + description_length_factor * 0.3
            
            # Clamp score between 0 and 1
            score = max(0.0, min(1.0, score))
            
            # Determine pass/fail and refinement need
            passed = score >= self.quality_threshold
            needs_refinement = score < self.refinement_threshold
            
            # Generate feedback based on score
            if score >= self.refinement_threshold:
                feedback = "Image successfully captures the prompt's requirements."
            elif score >= self.quality_threshold:
                feedback = "Image meets basic requirements but could be improved."
            else:
                feedback = "Image does not adequately match the prompt requirements."
            
            return {
                'score': score,
                'passed': passed,
                'needs_refinement': needs_refinement,
                'feedback': feedback
            }
            
        except Exception as e:
            logger.error(f"Error grading evaluation: {str(e)}")
            return {
                'score': 0.0,
                'passed': False,
                'needs_refinement': True,
                'feedback': f"Error during evaluation: {str(e)}"
            }

    async def setup(self):
        """Initialize any required resources."""
        pass

    async def cleanup(self):
        """Cleanup any resources."""
        pass

    async def __aenter__(self):
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup() 