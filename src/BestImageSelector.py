import logging
from typing import List, Dict, Optional
from pathlib import Path
import sqlite3
from config import DATABASE_PATH, QUALITY_THRESHOLD

logger = logging.getLogger(__name__)

class BestImageSelector:
    """Handles selection of best image variants based on evaluation scores."""

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path

    async def select_best_variant(self, variants: List[Dict]) -> Optional[Dict]:
        """
        Select the best variant from a list based on evaluation scores.
        
        Args:
            variants: List of variant dictionaries with evaluation scores
            
        Returns:
            Best variant or None if no variants pass quality threshold
        """
        if not variants:
            logger.warning("No variants provided for selection")
            return None
            
        # Filter variants that meet quality threshold
        qualified_variants = [
            v for v in variants 
            if v.get('evaluation_score', 0) >= QUALITY_THRESHOLD
        ]
        
        if not qualified_variants:
            logger.info("No variants met quality threshold")
            return None
            
        # Select variant with highest score
        best_variant = max(qualified_variants, 
                          key=lambda x: x.get('evaluation_score', 0))
        
        logger.info(f"Selected best variant with score {best_variant.get('evaluation_score')}")
        return best_variant

    async def save_best_variant(self, prompt_id: str, iteration: int, 
                              best_variant: Dict) -> bool:
        """Save the best variant to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO best_images 
                    (prompt_id, iteration, best_image_id, evaluation_score)
                    VALUES (?, ?, ?, ?)
                """, (
                    prompt_id,
                    iteration,
                    best_variant.get('id'),
                    best_variant.get('evaluation_score')
                ))
                logger.info(f"Saved best variant for {prompt_id}, iteration {iteration}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving best variant: {str(e)}")
            return False

    async def get_best_variant(self, prompt_id: str, iteration: int) -> Optional[Dict]:
        """Retrieve the best variant for a given prompt and iteration."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT bi.*, gi.image_path, gi.prompt_text, gi.variant
                    FROM best_images bi
                    JOIN generated_images gi ON bi.best_image_id = gi.id
                    WHERE bi.prompt_id = ? AND bi.iteration = ?
                """, (prompt_id, iteration))
                
                result = cursor.fetchone()
                if result:
                    return {
                        'id': result[0],
                        'prompt_id': result[1],
                        'iteration': result[2],
                        'image_path': result[3],
                        'evaluation_score': result[4],
                        'prompt_text': result[5],
                        'variant': result[6]
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving best variant: {str(e)}")
            return None

    async def setup(self):
        """Initialize resources if needed."""
        pass

    async def cleanup(self):
        """Cleanup resources if needed."""
        pass

    async def __aenter__(self):
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup() 