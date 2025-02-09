import asyncio
import logging
from pathlib import Path
from typing import Optional

from src.image_generator import ImageGenerator
from src.prompt_handler import PromptHandler
from src.api_client import FAL_AI_API_KEY, GEMINI_API_KEY, OUTPUT_BASE_PATH, INPUT_FILE_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def process_prompt(
    image_generator: ImageGenerator,
    prompt_handler: PromptHandler,
    prompt_id: str,
    prompt_data: dict
) -> None:
    """Process a single prompt through the image generation pipeline."""
    try:
        # Construct full prompt from components
        full_prompt = f"Title: {prompt_data['title']}\nScene: {prompt_data['scene']}\n" \
                     f"Mood: {prompt_data['mood']}\nPrompt: {prompt_data['prompt']}"

        # Generate first image
        logger.info(f"Generating initial image for prompt {prompt_id}")
        image_path = await image_generator.generate_image(full_prompt, prompt_id, iteration=1)
        if not image_path:
            logger.error(f"Failed to generate initial image for prompt {prompt_id}")
            return

        # Save first iteration results
        prompt_handler.save_results(
            prompt_id=prompt_id,
            iteration=1,
            image_path=image_path,
            prompt=full_prompt,
            evaluation=None
        )

        # Evaluate the image
        logger.info(f"Evaluating image for prompt {prompt_id}")
        evaluation = await image_generator.evaluate_image(image_path)
        if not evaluation:
            logger.error(f"Failed to evaluate image for prompt {prompt_id}")
            return

        # Generate refined prompt
        logger.info(f"Generating refined prompt for {prompt_id}")
        refined_prompt = await image_generator.refine_prompt(full_prompt, evaluation)
        if not refined_prompt:
            logger.error(f"Failed to generate refined prompt for {prompt_id}")
            return

        # Generate second image with refined prompt
        logger.info(f"Generating refined image for prompt {prompt_id}")
        refined_image_path = await image_generator.generate_image(refined_prompt, prompt_id, iteration=2)

        # Save second iteration results
        prompt_handler.save_results(
            prompt_id=prompt_id,
            iteration=2,
            image_path=refined_image_path,
            prompt=refined_prompt,
            evaluation=evaluation
        )

    except Exception as e:
        logger.error(f"Error processing prompt {prompt_id}: {str(e)}")

async def main():
    """Main function to orchestrate the image generation pipeline."""
    try:
        # Initialize components
        prompt_handler = PromptHandler(INPUT_FILE_PATH, OUTPUT_BASE_PATH)
        image_generator = ImageGenerator(
            fal_api_key=FAL_AI_API_KEY,
            gemini_api_key=GEMINI_API_KEY,
            output_base_path=OUTPUT_BASE_PATH
        )

        # Load prompts
        prompts = prompt_handler.load_prompts()
        if not prompts:
            logger.error("No prompts found. Exiting.")
            return

        logger.info(f"Processing {len(prompts)} prompts")

        # Process prompts in batches
        batch_size = image_generator.BATCH_SIZE
        for i in range(0, len(prompts), batch_size):
            batch_prompts = dict(list(prompts.items())[i:i + batch_size])
            
            # Process batch
            tasks = []
            for prompt_id, prompt_data in batch_prompts.items():
                task = process_prompt(
                    image_generator,
                    prompt_handler,
                    prompt_id,
                    prompt_data
                )
                tasks.append(task)

            # Run batch tasks concurrently
            await asyncio.gather(*tasks)
            logger.info(f"Completed batch {i//batch_size + 1}")

        logger.info("Completed processing all prompts")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
    finally:
        # Cleanup
        await image_generator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())