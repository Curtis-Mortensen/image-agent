"""
Core pipeline orchestrator that manages the complete image generation workflow.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, List, Callable
from PIL import Image
from rich.progress import Progress, TaskID
from concurrent.futures import ProcessPoolExecutor
import aiofiles

from src.BatchGenerator import BatchGenerator
from src.ImageVision import ImageVision
from src.EvaluationGrader import EvaluationGrader
from src.BestImageSelector import BestImageSelector
from src.PromptRefiner import PromptRefiner
from src.PromptHandler import PromptHandler
from src.DatabaseGenerator import DatabaseGenerator
from config import PIPELINE_CONFIG, DATABASE_PATH

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """Orchestrates the multi-variant image generation pipeline."""

    def __init__(self, 
                 input_file_path: Path,
                 output_base_path: Path,
                 fal_api_key: str,
                 gemini_api_key: str,
                 progress: Optional[Progress] = None):
        # Initialize core components
        self.output_base_path = Path(output_base_path)
        self.batch_generator = BatchGenerator(fal_api_key, output_base_path)
        self.image_vision = ImageVision(gemini_api_key)
        self.evaluation_grader = EvaluationGrader()
        self.best_selector = BestImageSelector()
        self.prompt_handler = PromptHandler(input_file_path, output_base_path)
        self.prompt_refiner = PromptRefiner(gemini_api_key, output_base_path, self.prompt_handler)
        
        # Initialize database
        self.db_generator = DatabaseGenerator()
        self.db_generator.initialize_database()
        
        # Setup progress tracking
        self.progress = progress
        self.running = True
        self.process_pool = ProcessPoolExecutor(
            max_workers=PIPELINE_CONFIG.get("max_workers", 2)
        )

    # Modular Component Operations
    async def generate_batch_variants(self, prompt_id: str, prompt_data: dict,
                                    iteration: int = 1, batch_size: Optional[int] = None,
                                    progress_task: Optional[TaskID] = None) -> List[Dict]:
        """Generate a batch of image variants."""
        if progress_task and self.progress:
            self.progress.update(progress_task, 
                               description=f"Generating variants for {prompt_id}")
        
        return await self.batch_generator.generate_batch(
            prompt_id, prompt_data, iteration, batch_size
        )

    async def evaluate_variants(self, variants: List[Dict], prompt: str,
                              progress_task: Optional[TaskID] = None) -> List[Dict]:
        """Get descriptions and grades for a list of variants."""
        if progress_task and self.progress:
            self.progress.update(progress_task, 
                               description="Evaluating variants")
        
        evaluated_variants = []
        for variant in variants:
            image = Image.open(variant['image_path'])
            description = await self.image_vision.describe_image(image)
            
            if description:
                evaluation = await self.evaluation_grader.grade_evaluation(
                    description, prompt
                )
                
                variant.update({
                    'description': description,
                    'evaluation_score': evaluation['score'],
                    'passed': evaluation['passed'],
                    'needs_refinement': evaluation['needs_refinement'],
                    'feedback': evaluation['feedback']
                })
                evaluated_variants.append(variant)
                
        return evaluated_variants

    async def select_best_variant(self, variants: List[Dict], prompt_id: str,
                                iteration: int) -> Optional[Dict]:
        """Select and save the best variant."""
        best_variant = await self.best_selector.select_best_variant(variants)
        if best_variant:
            await self.best_selector.save_best_variant(
                prompt_id, iteration, best_variant
            )
        return best_variant

    async def process_variants(self, prompt_id: str, prompt_data: dict,
                             iteration: int, batch_size: Optional[int] = None,
                             progress_task: Optional[TaskID] = None) -> Optional[Dict]:
        """Process a batch of variants for a single prompt iteration."""
        try:
            # 1. Generate batch of variants
            variants = await self.generate_batch_variants(
                prompt_id, prompt_data, iteration, batch_size, progress_task
            )
            
            if not variants:
                logger.error(f"No variants generated for {prompt_id}")
                return None

            # 2. Get descriptions and evaluate all variants
            evaluated_variants = await self.evaluate_variants(
                variants, prompt_data['prompt'], progress_task
            )
            
            if not evaluated_variants:
                logger.error(f"No variants could be evaluated for {prompt_id}")
                return None

            # 3. Select best variant
            best_variant = await self.select_best_variant(
                evaluated_variants, prompt_id, iteration
            )
            
            if best_variant:
                # 4. Refine prompt if needed and not final iteration
                if (iteration < PIPELINE_CONFIG["max_iterations"] and 
                    best_variant['needs_refinement']):
                    if progress_task and self.progress:
                        self.progress.update(progress_task, 
                                           description=f"Refining prompt for {prompt_id}")
                    
                    refined_prompt = await self.prompt_refiner.refine_prompt(
                        prompt_data['prompt'],
                        prompt_id,
                        {
                            'evaluation_text': best_variant['description'],
                            'feedback': best_variant['feedback']
                        }
                    )
                    if refined_prompt:
                        prompt_data['prompt'] = refined_prompt

            return best_variant

        except Exception as e:
            logger.error(f"Error processing variants for {prompt_id}: {str(e)}")
            return None

    async def run_pipeline(self, batch_only: bool = False) -> None:
        """Run the complete pipeline for all prompts."""
        try:
            async with (self.prompt_handler as ph,
                       self.batch_generator as bg,
                       self.image_vision as iv,
                       self.evaluation_grader as eg,
                       self.best_selector as bs,
                       self.prompt_refiner as pr):
                
                prompts = await self.prompt_handler.load_prompts()
                if not prompts:
                    logger.error("No prompts found")
                    return

                max_iterations = 1 if batch_only else PIPELINE_CONFIG["max_iterations"]
                logger.info(f"Starting pipeline for {len(prompts)} prompts, "
                          f"max {max_iterations} iterations")

                for prompt_id, prompt_data in prompts.items():
                    if not self.running:
                        break

                    task = None
                    if self.progress:
                        task = self.progress.add_task(
                            f"Processing {prompt_id}",
                            total=max_iterations
                        )

                    for iteration in range(1, max_iterations + 1):
                        best_variant = await self.process_variants(
                            prompt_id, prompt_data, iteration, PIPELINE_CONFIG["batch_size"]["default"], task
                        )
                        
                        if not best_variant:
                            logger.warning(f"No successful variants for {prompt_id} "
                                         f"iteration {iteration}")
                            break
                            
                        if self.progress:
                            self.progress.update(task, advance=1)
                            
                        # Stop if we got a good result and refinement isn't needed
                        if not best_variant.get('needs_refinement', True):
                            logger.info(f"Got satisfactory result for {prompt_id} "
                                      f"at iteration {iteration}")
                            break

                logger.info("Pipeline completed successfully")

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
        finally:
            self.process_pool.shutdown(wait=True)

    async def evaluate_single_image(self, image_path: Path) -> Optional[Dict]:
        """Evaluate a single image on demand."""
        try:
            image = Image.open(image_path)
            description = await self.image_vision.describe_image(image)
            
            if description:
                # Save description to file
                eval_path = image_path.parent / f"{image_path.stem}_description.txt"
                async with aiofiles.open(eval_path, 'w') as f:
                    await f.write(description)
                
                return {
                    'description': description,
                    'description_path': str(eval_path)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error evaluating image: {str(e)}")
            return None

    async def refine_single_prompt(self, prompt: str, description: str) -> Optional[str]:
        """Refine a single prompt on demand."""
        try:
            evaluation = await self.evaluation_grader.grade_evaluation(
                description, prompt
            )
            
            if evaluation['needs_refinement']:
                refined = await self.prompt_refiner.refine_prompt(
                    prompt, "manual_prompt", 
                    {'evaluation_text': description, 'feedback': evaluation['feedback']}
                )
                return refined
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error refining prompt: {str(e)}")
            return None

    async def setup(self):
        """Initialize resources."""
        pass

    async def cleanup(self):
        """Cleanup resources."""
        pass

    async def __aenter__(self):
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup() 