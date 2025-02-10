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

from src.image_generator import ImageGenerator # Import ImageGenerator
from src.prompt_handler import PromptHandler # Import PromptHandler
from src.image_evaluator import ImageEvaluator
from src.prompt_refiner import PromptRefiner
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
        self.image_generator = ImageGenerator(
            fal_api_key=fal_api_key,
            gemini_api_key=gemini_api_key,
            output_base_path=output_base_path,
            prompt_handler=self.prompt_handler
        )
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

    async def process_prompt(self, prompt_id: str, prompt_data: dict,
                           progress_task: TaskID) -> bool:
        """Process a single prompt through the generation pipeline."""
        try:
            def update_progress(message: str):
                self.progress.update(progress_task, description=message)
            # Delegate prompt processing to ImageGenerator
            return await self.image_generator.process_prompt(
                prompt_id,
                prompt_data,
                update_progress # Pass the progress update callback
            )

        except asyncio.CancelledError:
            logger.warning(f"Processing cancelled for prompt {prompt_id}")
            raise
        except Exception as e:
            logger.error(f"Error processing prompt {prompt_id}: {str(e)}", exc_info=True)
            return False

    async def process_batch(self, batch_prompts: Dict[str, dict]) -> None:
        """Process a batch of prompts concurrently."""
        tasks = []
        with self.progress:
            overall_progress = self.progress.add_task(
                "Processing prompts...",
                total=len(batch_prompts)
            )

            for prompt_id, prompt_data in batch_prompts.items():
                if not self.running:
                    break
                task = asyncio.create_task(
                    self.process_prompt(prompt_id, prompt_data, overall_progress)
                )
                tasks.append(task)
                self.tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            self.tasks = [t for t in self.tasks if not t.done()]

        successful = sum(1 for r in results if r is True)
        logger.info(f"Batch completed: {successful}/{len(results)} successful")

    async def run_image_generation(self) -> None:
        """Run the complete image generation."""
        try:
            async with self.prompt_handler, self.image_generator:
                prompts = await self.prompt_handler.load_prompts()
                if not prompts:
                    logger.error("No prompts found. Exiting.")
                    return

                total_prompts = len(prompts)
                logger.info(f"Starting Image Generation for {total_prompts} prompts")

                # Directly process all prompts without batching for now, to ensure all are processed
                for prompt_id, prompt_data in prompts.items():
                    if not self.running:
                        break
                    logger.info(f"Processing prompt: {prompt_id}")
                    await self.process_prompt(prompt_id, prompt_data, self.progress.add_task(f"Prompt {prompt_id}", total=2)) # Assuming 2 iterations per prompt

                logger.info("Image Generation completed successfully")

        except asyncio.CancelledError:
            logger.info("Image Generation cancelled - starting cleanup")
        except Exception as e:
            logger.error(f"Image Generation failed: {str(e)}", exc_info=True)
        finally:
            self.process_pool.shutdown(wait=True)

# --- Interactive Commands ---
async def generate_images_command():
    """Executes the full image generation pipeline."""
    pipeline = ImageGenerationPipeline(
        input_file_path=INPUT_FILE_PATH,
        output_base_path=OUTPUT_BASE_PATH,
        fal_api_key=FAL_KEY,
        gemini_api_key=GEMINI_API_KEY
    )
    await pipeline.run_image_generation()

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

    evaluator = ImageEvaluator(GEMINI_API_KEY)
    await evaluator.setup()
    try:
        evaluation_result = await evaluator.evaluate_image(Image.open(image_path))
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
        await evaluator.cleanup()

async def refine_prompt_command():
    """Refines a prompt using Gemini API based on evaluation of an image."""
    original_prompt = input("Enter the original prompt you want to refine: ")
    evaluation_text = input("Enter the evaluation text for the generated image: ")

    refiner = PromptRefiner(GEMINI_API_KEY)
    await refiner.setup()
    try:
        refined_prompt = await refiner.refine_prompt(original_prompt, {'evaluation_text': evaluation_text})
        if refined_prompt:
            print("\nRefined Prompt:")
            print(refined_prompt)
        else:
            print("Prompt refinement failed.")
    finally:
        await refiner.cleanup()

async def main_menu():
    """Displays the main menu and handles user input."""
    while True:
        print("\nChoose an action:")
        print("1. Run Image Generation")
        print("2. Evaluate an Existing Image")
        print("3. Refine a Prompt based on Evaluation")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            await generate_images_command()
        elif choice == '2':
            await evaluate_image_command()
        elif choice == '3':
            await refine_prompt_command()
        elif choice == '4':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

@click.command()
def main_cli():
    """Runs the interactive command line interface."""
    asyncio.run(main_menu())

if __name__ == '__main__':
    main_cli()
