import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, Callable, List
import fal_client
import aiohttp
from config import DATABASE_PATH, PIPELINE_CONFIG
from src.APIClient import FalClient
import aiofiles

logger = logging.getLogger(__name__)

class BatchGenerator:
    """Handles batch generation of image variants."""
    
    def __init__(self, fal_key: str, output_base_path: Path):
        """Initialize the batch generator."""
        self.fal_client = FalClient(fal_key)
        self.output_base_path = Path(output_base_path)
        self.batch_config = PIPELINE_CONFIG["batch_size"]
        
    def _get_batch_size(self, requested_size: Optional[int] = None) -> int:
        """Get the batch size within configured limits."""
        if requested_size is None:
            return self.batch_config['default']
            
        return max(
            self.batch_config['min'],
            min(requested_size, self.batch_config['max'])
        )

    async def generate_batch(self, prompt_id: str, prompt_data: dict, 
                           iteration: int = 1, batch_size: Optional[int] = None) -> List[Dict]:
        """Generate a batch of image variants."""
        try:
            actual_batch_size = self._get_batch_size(batch_size)
            logger.info(f"Generating batch of {actual_batch_size} variants for {prompt_id}")
            
            variants = []
            for i in range(actual_batch_size):
                variant_path = self.output_base_path / f"{prompt_id}_iter{iteration}_var{i+1}.png"
                
                try:
                    image_data = await self.fal_client.generate_image(prompt_data['prompt'])
                    if image_data:
                        async with aiofiles.open(variant_path, 'wb') as f:
                            await f.write(image_data)
                        
                        variants.append({
                            'image_path': str(variant_path),
                            'variant': i + 1,
                            'prompt_id': prompt_id,
                            'iteration': iteration
                        })
                        
                except Exception as e:
                    logger.error(f"Error generating variant {i+1}: {str(e)}")
                    continue
                    
            return variants
            
        except Exception as e:
            logger.error(f"Error in generate_batch: {str(e)}")
            return []

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass 