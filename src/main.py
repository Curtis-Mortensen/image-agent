import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import signal
from datetime import datetime
import click
from rich.console import Console # Removed rich console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID # Removed rich progress
from rich.logging import RichHandler # Removed rich handler
import aiofiles.os
from concurrent.futures import ProcessPoolExecutor

from src.image_generator import ImageGenerator
from src.prompt_handler import PromptHandler

# Configure rich console - REMOVED
# console = Console()

# Configure logging with rich - SIMPLIFIED
logging.basicConfig(
    level=logging.DEBUG, # Set level to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s', # Basic format
    handlers=[
        logging.StreamHandler(), # Basic console handler
        logging.FileHandler(f"generation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

class ImageGenerationPipeline:
    """Orchestrates the image generation pipeline with progress tracking."""

    def __init__(self, input_file_path: Path, output_base_path: Path,
                 fal_api_key: str, gemini_api_key: str):
        """Initialize the pipeline with rich progress tracking."""
        self.prompt_handler = PromptHandler(input_file_path, output_base_path)
        self.image_generator = ImageGenerator(
            fal_api_key=fal_api_key,
            gemini_api_key=gemini_api_key,
            output_base_path=output_base_path
        )
        self.running = True
        self.tasks: List[asyncio.Task] = []
        # Rich progress - REMOVED for simplification
        # self.progress = Progress(
        #     SpinnerColumn(),
        #     TextColumn("[progress.description]{task.description}"),
        #     BarColumn(),
        #     TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        #     console=console
        # )
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
                           progress_task: TaskID) -> bool: # Removed progress_task: TaskID
        """Process a single prompt through the generation pipeline."""
        try:
            # def update_progress(message: str): # Removed progress update
            #     self.progress.update(progress_task, description=message)

            # Generate and evaluate image
            image_path, evaluation = await self.image_generator.generate_and_evaluate(
                prompt_data,
                prompt_id,
                # progress_callback=update_progress # Removed progress callback
            )

            if not image_path:
                return False

            # Save results
            await self.prompt_handler.save_results(
                prompt_id=prompt_id,
                iteration=2,  # Final iteration
                image_path=image_path,
                prompt=prompt_data['prompt'],
                evaluation=evaluation
            )

            # self.progress.update(progress_task, advance=1) # Removed progress update
            logger.info(f"Successfully completed processing for prompt {prompt_id}")
            return True

        except asyncio.CancelledError:
            logger.warning(f"Processing cancelled for prompt {prompt_id}")
            raise
        except Exception as e:
            logger.error(f"Error processing prompt {prompt_id}: {str(e)}", exc_info=True)
            return False

    async def process_batch(self, batch_prompts: Dict[str, dict]) -> None:
        """Process a batch of prompts concurrently."""
        tasks = []
        # with self.progress: # Removed rich progress
        #     overall_progress = self.progress.add_task(
        #         "Processing prompts...",
        #         total=len(batch_prompts)
        #     )

        for prompt_id, prompt_data in batch_prompts.items():
            if not self.running:
                break
            task = asyncio.create_task(
                self.process_prompt(prompt_id, prompt_data, None) # Removed overall_progress
            )
            tasks.append(task)
            self.tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        self.tasks = [t for t in self.tasks if not t.done()]

        successful = sum(1 for r in results if r is True)
        logger.info(f"Batch completed: {successful}/{len(results)} successful")

    async def run(self) -> None:
        """Run the complete pipeline."""
        logger.info("Starting pipeline run...") # ADDED LOGGING START
        try:
            async with self.prompt_handler, self.image_generator:
                prompts = await self.prompt_handler.load_prompts()
                if not prompts:
                    logger.error("No prompts found. Exiting.")
                    return

                total_prompts = len(prompts)
                logger.info(f"Starting processing of {total_prompts} prompts")

                batch_size = self.image_generator.batch_size
                for i in range(0, total_prompts, batch_size):
                    if not self.running:
                        break

                    batch_prompts = dict(list(prompts.items())[i:i + batch_size])
                    logger.info(f"Processing batch {i//batch_size + 1}")
                    await self.process_batch(batch_prompts)

                logger.info("Pipeline completed successfully")

        except asyncio.CancelledError:
            logger.info("Pipeline cancelled - starting cleanup")
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        finally:
            self.process_pool.shutdown(wait=True)
        logger.info("Pipeline run finished.") # ADDED LOGGING END

@click.command()
@click.option('--input-file', type=click.Path(exists=True), help='Path to input prompts JSON file')
@click.option('--output-dir', type=click.Path(), help='Path to output directory')
@click.option('--batch-size', type=int, default=3, help='Number of concurrent generations')
def main(input_file: Optional[str], output_dir: Optional[str], batch_size: int):
    """Entry point for the application."""
    try:
        from config import FAL_KEY, GEMINI_API_KEY, OUTPUT_BASE_PATH, INPUT_FILE_PATH

        input_path = Path(input_file) if input_file else INPUT_FILE_PATH
        output_path = Path(output_dir) if output_dir else OUTPUT_BASE_PATH

        pipeline = ImageGenerationPipeline(
            input_file_path=input_path,
            output_base_path=output_path,
            fal_api_key=FAL_KEY,
            gemini_api_key=GEMINI_API_KEY
        )

        pipeline.setup_signal_handlers()
        asyncio.run(pipeline.run())

    except Exception as e:
        logger.error(f"Application failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
