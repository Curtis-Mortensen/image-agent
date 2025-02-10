import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List # Explicitly include List and Dict
import signal
import click
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.logging import RichHandler
from PIL import Image
import aiofiles # Import aiofiles for async file operations
from concurrent.futures import ProcessPoolExecutor # Import ProcessPoolExecutor

from src.image_generator import ImageGenerator # Import ImageGenerator
from src.prompt_handler import PromptHandler # Import PromptHandler
from src.image_evaluator import ImageEvaluator # Import ImageEvaluator
from src.prompt_refiner import PromptRefiner # Import PromptRefiner
from config import FAL_KEY, GEMINI_API_KEY, INPUT_FILE_PATH, OUTPUT_BASE_PATH

# Configure rich console
console = Console()

# Configure logging with rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(console=console, rich_tracebacks=True),
        logging.FileHandler(f"generation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

class ImageGenerationPipeline:
    """Orchestrates the image generation pipeline."""

    def __init__(self, input_file_path: Path, output_base_path: Path,
                 fal_api_key: str, gemini_api_key: str):
        """Initialize the pipeline with rich progress tracking."""
        self.prompt_handler = PromptHandler(input_file_path, output_base_path)
        self.image_generator = ImageGenerator( # Removed Gemini and PromptHandler deps
            fal_api_key=fal_api_key,
            output_base_path=output_base_path,
        )
        self.image_evaluator = ImageEvaluator(gemini_api_key) # Initialize here
        self.prompt_refiner = PromptRefiner(gemini_api_key, output_base_path, self.prompt_handler) # Initialize here
        self.running = True
        self.tasks: List[asyncio.Task] = []
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        )
        self.process_pool = ProcessPoolExecutor(max_workers=2)

    def setup_signal_handlers(self):
        """Set up graceful shutdown handlers."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals with cleanup."""
        logger.info(f"Received signal {signum}. Starting graceful shutdown...")
        self.running = False
        for task in self.tasks:
            task.cancel()
        self.process_pool.shutdown(wait=False)

    async def generate_image_for_prompt(self, prompt_id: str, prompt_data: dict, iteration: int, progress_task: TaskID) -> Optional[Path]: # New function for just generation
        """Generate a single image iteration and return image path."""
        try:
            def update_progress(message: str):
                self.progress.update(progress_task, description=message)
            # Delegate prompt processing to ImageGenerator
            image_path = await self.image_generator.process_prompt(
                prompt_id,
                prompt_data,
                iteration=iteration,
                progress_callback=update_progress # Pass the progress update callback
            )
            return image_path

        except asyncio.CancelledError:
            logger.warning(f"Processing cancelled during image generation for prompt {prompt_id}")
            raise
        except Exception as e:
            logger.error(f"Error during image generation for prompt {prompt_id}: {str(e)}", exc_info=True)
            return None

    async def evaluate_generated_image(self, prompt_id: str, image_path: Path, progress_task: TaskID) -> Optional[Dict]: # New function for evaluation
        """Evaluate a generated image and return evaluation results."""
        try:
            def update_progress(message: str):
                self.progress.update(progress_task, description=message)
            self.progress.update(progress_task, description=f"Evaluating image for {prompt_id}")
            evaluation = await self.image_evaluator.evaluate_image(Image.open(image_path))
            return evaluation

        except asyncio.CancelledError:
            logger.warning(f"Evaluation cancelled for prompt {prompt_id}")
            raise
        except Exception as e:
            logger.error(f"Error during image evaluation for prompt {prompt_id}: {str(e)}", exc_info=True)
            return None

    async def refine_prompt_based_on_evaluation(self, prompt_id: str, original_prompt: str, evaluation: Dict, progress_task: TaskID) -> Optional[str]: # New function for prompt refinement
        """Refine a prompt based on image evaluation and return refined prompt."""
        try:
            def update_progress(message: str):
                self.progress.update(progress_task, description=message)
            self.progress.update(progress_task, description=f"Refining prompt for {prompt_id}")
            refined_prompt = await self.prompt_refiner.refine_prompt(original_prompt, prompt_id, evaluation)
            return refined_prompt

        except asyncio.CancelledError:
            logger.warning(f"Prompt refinement cancelled for prompt {prompt_id}")
            raise
        except Exception as e:
            logger.error(f"Error during prompt refinement for prompt {prompt_id}: {str(e)}", exc_info=True)
            return None


    async def process_iteration(self, prompt_id: str, prompt_data: dict, iteration: int, progress_task: TaskID) -> bool: # Process single iteration (generate and save results)
        """Process a single iteration: generate image and save results."""
        try:
            def update_progress(message: str):
                self.progress.update(progress_task, description=message)

            # Generate image
            image_path = await self.generate_image_for_prompt(prompt_id, prompt_data, iteration, progress_task) # Call generate_image_for_prompt
            if not image_path:
                return False

            # Save results - use PromptHandler instance
            if self.prompt_handler:
                await self.prompt_handler.save_results(
                    prompt_id=prompt_id,
                    iteration=iteration,
                    image_path=image_path,
                    prompt=prompt_data['prompt'], # Or get full prompt data if needed
                    evaluation=None # No evaluation at this stage
                )
            else:
                logger.error("PromptHandler not initialized in ImageGenerationPipeline. Cannot save results.")
                return False
            return True

        except asyncio.CancelledError:
            logger.warning(f"Processing cancelled for prompt {prompt_id}, iteration {iteration}")
            raise
        except Exception as e:
            logger.error(f"Error processing prompt {prompt_id}, iteration {iteration}: {str(e)}", exc_info=True)
            return False


    async def run_image_generation(self) -> None: # Run only image generation (iteration 1 and 2)
        """Run the complete image generation for two iterations."""
        try:
            async with self.prompt_handler, self.image_generator, self.image_evaluator, self.prompt_refiner: # Keep context managers for all
                prompts = await self.prompt_handler.load_prompts()
                if not prompts:
                    logger.error("No prompts found. Exiting.")
                    return

                total_prompts = len(prompts)
                logger.info(f"Starting Image Generation for {total_prompts} prompts, 2 iterations each")

                for prompt_id, prompt_data in prompts.items():
                    if not self.running:
                        break
                    logger.info(f"Processing prompt: {prompt_id}")
                    prompt_progress_task = self.progress.add_task(f"Prompt {prompt_id}", total=2) # Two iterations
                    for iteration in range(1, 3): # Run two iterations
                        if not await self.process_iteration(prompt_id, prompt_data, iteration, prompt_progress_task): # Generate each iteration
                            logger.error(f"Iteration {iteration} failed for prompt {prompt_id}. Stopping prompt processing.")
                            break # Stop iterations for this prompt if one fails
                        self.progress.update(prompt_progress_task, advance=1) # Advance progress bar

                logger.info("Image Generation completed successfully for all prompts and iterations.")

        except asyncio.CancelledError:
            logger.info("Image Generation cancelled - starting cleanup")
        except Exception as e:
            logger.error(f"Image Generation failed: {str(e)}", exc_info=True)
        finally:
            self.process_pool.shutdown(wait=True)

    async def run_full_pipeline(self) -> None:
        """Runs the full image generation, evaluation, and refinement pipeline (up to 3 iterations)."""
        try:
            async with self.prompt_handler, self.image_generator, self.image_evaluator, self.prompt_refiner:
                prompts = await self.prompt_handler.load_prompts()
                if not prompts:
                    logger.error("No prompts found. Exiting.")
                    return

                total_prompts = len(prompts)
                logger.info(f"Starting Full Image Generation Pipeline for {total_prompts} prompts, up to 3 iterations each")

                for prompt_id, prompt_data in prompts.items():
                    if not self.running:
                        break
                    logger.info(f"Processing prompt: {prompt_id}")
                    prompt_progress_task = self.progress.add_task(f"Pipeline for Prompt {prompt_id}", total=3) # Up to 3 iterations in pipeline
                    current_prompt_data = prompt_data # Start with initial prompt data

                    for iteration in range(1, 4): # Run up to 3 iterations for full pipeline
                        logger.info(f"Starting iteration {iteration} for prompt {prompt_id}")

                        # 1. Generate Image
                        image_path = await self.generate_image_for_prompt(prompt_id, current_prompt_data, iteration, prompt_progress_task)
                        if not image_path:
                            logger.error(f"Image generation failed for prompt {prompt_id}, iteration {iteration}. Stopping pipeline for this prompt.")
                            break

                        # 2. Evaluate Image
                        evaluation = await self.evaluate_generated_image(prompt_id, image_path, prompt_progress_task)
                        if not evaluation:
                            logger.error(f"Image evaluation failed for prompt {prompt_id}, iteration {iteration}. Stopping pipeline for this prompt.")
                            break

                        # 3. Refine Prompt (for iterations 1 and 2)
                        if iteration < 3: # Refine only for first 2 iterations
                            refined_prompt = await self.refine_prompt_based_on_evaluation(prompt_id, current_prompt_data['prompt'], evaluation, prompt_progress_task)
                            if refined_prompt and "No refinement needed" not in refined_prompt:
                                current_prompt_data['prompt'] = refined_prompt # Update prompt data with refined prompt for next iteration
                                logger.info(f"Prompt refined for iteration {iteration+1}: {refined_prompt}")
                            else:
                                logger.info(f"No prompt refinement needed or refinement failed for iteration {iteration}.")
                        else:
                            logger.info(f"No prompt refinement in the final iteration {iteration}.")


                        # 4. Save Results (after each iteration) - including evaluation and potentially refined prompt
                        if self.prompt_handler:
                            refined_prompt_to_save = current_prompt_data['prompt'] if iteration < 3 and refined_prompt and "No refinement needed" not in refined_prompt else None # Save refined prompt only when it was actually refined
                            await self.prompt_handler.save_results(
                                prompt_id=prompt_id,
                                iteration=iteration,
                                image_path=image_path,
                                prompt=current_prompt_data['prompt'], # Save current prompt (refined or original)
                                evaluation=evaluation # Save evaluation result
                            )
                        else:
                            logger.error("PromptHandler not initialized in ImageGenerationPipeline. Cannot save results.")
                            break # Stop if saving results fails

                        self.progress.update(prompt_progress_task, advance=1) # Advance progress bar for each iteration step

                    logger.info(f"Full pipeline completed for prompt {prompt_id}.")

                logger.info("Full Image Generation Pipeline completed successfully for all prompts.")

        except asyncio.CancelledError:
            logger.info("Full Image Generation Pipeline cancelled - starting cleanup")
        except Exception as e:
            logger.error(f"Full Image Generation Pipeline failed: {str(e)}", exc_info=True)
        finally:
            self.process_pool.shutdown(wait=True)


# --- Interactive Commands ---
async def generate_images_command():
    """Executes the image generation pipeline (2 iterations)."""
    pipeline = ImageGenerationPipeline(
        input_file_path=INPUT_FILE_PATH,
        output_base_path=OUTPUT_BASE_PATH,
        fal_api_key=FAL_KEY,
        gemini_api_key=GEMINI_API_KEY
    )
    await pipeline.run_image_generation() # Call run_image_generation for just generation

async def evaluate_image_command():
    """Evaluates an existing image using Gemini API, saving results to outputs/evaluations."""
    default_image_path = OUTPUT_BASE_PATH / "images" / "scene_001_iteration_2.png" # Updated default image path - no scene_001 subfolder
    image_path_str = input(f"Enter the path to the image to evaluate (default: {default_image_path}): ")

    if not image_path_str:
        image_path = default_image_path
    else:
        image_path = Path(image_path_str)

    if not image_path.exists() or not image_path.is_file():
        print("Invalid image path. Please provide a valid file path.")
        return

    pipeline = ImageGenerationPipeline( # Need pipeline to access evaluator
        input_file_path=INPUT_FILE_PATH,
        output_base_path=OUTPUT_BASE_PATH,
        fal_api_key=FAL_KEY,
        gemini_api_key=GEMINI_API_KEY
    )
    await pipeline.image_evaluator.setup() # Setup evaluator explicitly
    try:
        evaluation_progress = pipeline.progress.add_task("Evaluating image...", total=1) # Add progress task
        evaluation_result = await pipeline.evaluate_generated_image("user_provided_image", image_path, evaluation_progress) # Use new evaluate function
        pipeline.progress.update(evaluation_progress, advance=1) # Complete progress

        if evaluation_result:
            print("\nEvaluation Result:")
            print(evaluation_result['evaluation_text'])

            # Save evaluation to outputs/evaluations - no scene_001 subfolder
            evaluation_dir = OUTPUT_BASE_PATH / "evaluations" # Updated evaluation dir - no scene_001 subfolder
            await aiofiles.os.makedirs(evaluation_dir, exist_ok=True) # Ensure dir exists
            evaluation_file_path = evaluation_dir / f"{image_path.name}.evaluation.txt" # Use image_path.name to keep filename
            async with aiofiles.open(evaluation_file_path, 'w') as f:
                await f.write(evaluation_result['evaluation_text'])
            print(f"Evaluation saved to: {evaluation_file_path}")

        else:
            print("Image evaluation failed.")
    finally:
        await pipeline.image_evaluator.cleanup() # Cleanup evaluator explicitly

async def refine_prompt_command():
    """Refines a prompt using Gemini API based on evaluation of an image."""
    original_prompt = input("Enter the original prompt you want to refine: ")
    evaluation_text = input("Enter the evaluation text for the generated image: ")

    pipeline = ImageGenerationPipeline( # Need pipeline to access refiner
        input_file_path=INPUT_FILE_PATH,
        output_base_path=OUTPUT_BASE_PATH,
        fal_api_key=FAL_KEY,
        gemini_api_key=GEMINI_API_KEY
    )
    await pipeline.prompt_refiner.setup() # Setup refiner explicitly
    try:
        refinement_progress = pipeline.progress.add_task("Refining prompt...", total=1) # Add progress task
        refined_prompt = await pipeline.prompt_refiner.refine_prompt(original_prompt, "user_prompt", {'evaluation_text': evaluation_text}) # Use refiner from pipeline
        pipeline.progress.update(refinement_progress, advance=1) # Complete progress
        if refined_prompt:
            print("\nRefined Prompt:")
            print(refined_prompt)
        else:
            print("Prompt refinement failed.")
    finally:
        await pipeline.prompt_refiner.cleanup() # Cleanup refiner explicitly

async def run_full_pipeline_command():
    """Executes the full image generation, evaluation, and refinement pipeline (up to 3 iterations)."""
    pipeline = ImageGenerationPipeline(
        input_file_path=INPUT_FILE_PATH,
        output_base_path=OUTPUT_BASE_PATH,
        fal_api_key=FAL_KEY,
        gemini_api_key=GEMINI_API_KEY
    )
    await pipeline.run_full_pipeline() # Call run_full_pipeline for the full process


async def main_menu():
    """Displays the main menu and handles user input."""
    while True:
        print("\nChoose an action:")
        print("1. Run Image Generation (2 iterations)")
        print("2. Evaluate an Existing Image")
        print("3. Refine a Prompt based on Evaluation")
        print("4. Run Full Pipeline (Generate, Evaluate, Refine - up to 3 iterations)") # Option 4 for full pipeline
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            await generate_images_command()
        elif choice == '2':
            await evaluate_image_command()
        elif choice == '3':
            await refine_prompt_command()
        elif choice == '4':
            await run_full_pipeline_command() # Call new full pipeline command
        elif choice == '5':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

@click.command()
def main_cli():
    """Runs the interactive command line interface."""
    asyncio.run(main_menu())

if __name__ == '__main__':
    main_cli()
