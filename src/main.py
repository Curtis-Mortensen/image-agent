import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import signal
from datetime import datetime

from src.image_generator import ImageGenerator
from src.prompt_handler import PromptHandler
from src.api_client import FalClient, GeminiClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"generation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

class ImageGenerationPipeline:
    """Orchestrates the image generation pipeline."""
    
    def __init__(self, input_file_path: Path, output_base_path: Path,
                 fal_api_key: str, gemini_api_key: str):
        """
        Initialize the pipeline.
        
        Args:
            input_file_path: Path to input prompts JSON
            output_base_path: Base path for output files
            fal_api_key: FAL.ai API key
            gemini_api_key: Google Gemini API key
        """
        self.prompt_handler = PromptHandler(input_file_path, output_base_path)
        self.image_generator = ImageGenerator(
            fal_api_key=fal_api_key,
            gemini_api_key=gemini_api_key,
            output_base_path=output_base_path
        )
        self.running = True
        self.tasks: List[asyncio.Task] = []

    def setup_signal_handlers(self):
        """Set up graceful shutdown handlers."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}. Starting graceful shutdown...")
        self.running = False
        for task in self.tasks:
            task.cancel()

    async def process_prompt(self, prompt_id: str, prompt_data: dict) -> bool:
        """
        Process a single prompt through the generation pipeline.
        
        Args:
            prompt_id: Unique identifier for the prompt
            prompt_data: Dictionary containing prompt data
            
        Returns:
            bool indicating success/failure
        """
        try:
            # Construct full prompt
            full_prompt = (
                f"Title: {prompt_data['title']}\n"
                f"Scene: {prompt_data['scene']}\n"
                f"Mood: {prompt_data['mood']}\n"
                f"Prompt: {prompt_data['prompt']}"
            )

            # First iteration
            logger.info(f"Starting first iteration for prompt {prompt_id}")
            image_path = await self.image_generator.generate_image(
                full_prompt, 
                prompt_id,
                iteration=1
            )
            if not image_path:
                logger.error(f"Failed to generate initial image for {prompt_id}")
                return False

            # Save first iteration results
            self.prompt_handler.save_results(
                prompt_id=prompt_id,
                iteration=1,
                image_path=image_path,
                prompt=full_prompt,
                evaluation=None
            )

            # Evaluate first iteration
            logger.info(f"Evaluating first iteration for {prompt_id}")
            evaluation = await self.image_generator.evaluate_image(image_path)
            if not evaluation:
                logger.error(f"Failed to evaluate image for {prompt_id}")
                return False

            # Generate refined prompt
            logger.info(f"Generating refined prompt for {prompt_id}")
            refined_prompt = await self.image_generator.refine_prompt(
                full_prompt,
                evaluation
            )
            if not refined_prompt:
                logger.error(f"Failed to generate refined prompt for {prompt_id}")
                return False

            # Second iteration
            logger.info(f"Starting second iteration for prompt {prompt_id}")
            refined_image_path = await self.image_generator.generate_image(
                refined_prompt,
                prompt_id,
                iteration=2
            )
            if not refined_image_path:
                logger.error(f"Failed to generate refined image for {prompt_id}")
                return False

            # Save second iteration results
            self.prompt_handler.save_results(
                prompt_id=prompt_id,
                iteration=2,
                image_path=refined_image_path,
                prompt=refined_prompt,
                evaluation=evaluation
            )

            logger.info(f"Successfully completed processing for prompt {prompt_id}")
            return True

        except asyncio.CancelledError:
            logger.warning(f"Processing cancelled for prompt {prompt_id}")
            raise
        except Exception as e:
            logger.error(f"Error processing prompt {prompt_id}: {str(e)}", exc_info=True)
            return False

    async def process_batch(self, batch_prompts: Dict[str, dict]) -> None:
        """
        Process a batch of prompts concurrently.
        
        Args:
            batch_prompts: Dictionary of prompt_id to prompt_data mappings
        """
        tasks = []
        for prompt_id, prompt_data in batch_prompts.items():
            if not self.running:
                break
            task = asyncio.create_task(
                self.process_prompt(prompt_id, prompt_data)
            )
            tasks.append(task)
            self.tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        self.tasks = [t for t in self.tasks if not t.done()]
        
        # Log batch results
        successful = sum(1 for r in results if r is True)
        logger.info(f"Batch completed: {successful}/{len(results)} successful")

    async def run(self) -> None:
        """Run the complete pipeline."""
        try:
            # Initialize components
            await self.image_generator.setup()
            
            # Load prompts
            prompts = self.prompt_handler.load_prompts()
            if not prompts:
                logger.error("No prompts found. Exiting.")
                return

            total_prompts = len(prompts)
            logger.info(f"Starting processing of {total_prompts} prompts")

            # Process in batches
            batch_size = self.image_generator.BATCH_SIZE
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
            # Cleanup
            await self.image_generator.cleanup()

async def main():
    """Entry point for the application."""
    try:
        # Load environment variables
        from src.api_client import FAL_AI_API_KEY, GEMINI_API_KEY
        from src.api_client import OUTPUT_BASE_PATH, INPUT_FILE_PATH
        
        # Initialize and run pipeline
        pipeline = ImageGenerationPipeline(
            input_file_path=Path(INPUT_FILE_PATH),
            output_base_path=Path(OUTPUT_BASE_PATH),
            fal_api_key=FAL_AI_API_KEY,
            gemini_api_key=GEMINI_API_KEY
        )
        
        pipeline.setup_signal_handlers()
        await pipeline.run()

    except Exception as e:
        logger.error(f"Application failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
