import asyncio
import json
from pathlib import Path
from src.BatchGenerator import BatchGenerator
from src.APIClient import FalClient
from config import FAL_KEY
import logging
from rich.logging import RichHandler
from rich.console import Console
import sys

# Set up logging with more detailed format
console = Console()
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RichHandler(console=console, rich_tracebacks=True),
        logging.FileHandler('test_fal.log')
    ]
)
logger = logging.getLogger(__name__)

async def main():
    try:
        logger.info("Starting FAL.ai test script")
        
        # Validate FAL_KEY
        if not FAL_KEY:
            logger.error("FAL_KEY is not set in config.py")
            return
        
        # Load the actual prompt
        logger.info("Loading prompt data...")
        try:
            with open('data/inputs/prompts.json') as f:
                prompts = json.load(f)
                prompt_data = next((p for p in prompts['prompts'] if p['id'] == 'scene_001'), None)
                
            if not prompt_data:
                logger.error("Could not find scene_001 prompt")
                return
                
            logger.info(f"Loaded prompt data: {prompt_data['id']}")
            
        except FileNotFoundError:
            logger.error("prompts.json file not found in data/inputs/")
            return
        except json.JSONDecodeError:
            logger.error("Invalid JSON in prompts.json")
            return
            
        # Initialize the batch generator
        output_path = Path('data/outputs')
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing BatchGenerator with output path: {output_path}")
        batch_generator = BatchGenerator(FAL_KEY, output_path)
        
        # Generate a single variant
        logger.info("Starting batch generation...")
        variants = await batch_generator.generate_batch('scene_001', prompt_data, iteration=1, batch_size=1)
        
        if variants:
            logger.info("Successfully generated variants:")
            for v in variants:
                logger.info(f"- {v['image_path']}")
                # Verify the image file exists
                if not Path(v['image_path']).exists():
                    logger.warning(f"Generated image file does not exist: {v['image_path']}")
        else:
            logger.error("No variants were generated")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test script interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1) 