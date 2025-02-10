import google.generativeai as genai
import logging
from typing import Optional, Dict
from PIL import Image
import sqlite3
from datetime import datetime
from config import DATABASE_PATH

logger = logging.getLogger(__name__)

class ImageEvaluator:
    """Handles image evaluation using Google Gemini API with database tracking."""

    def __init__(self, api_key: str, db_path: str = DATABASE_PATH):
        self.api_key = api_key
        self.db_path = db_path
        
        # Initialize Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize database table for evaluation tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS image_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    evaluation_text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'completed',
                    needs_refinement BOOLEAN DEFAULT FALSE,
                    UNIQUE(image_path)
                )
            """)

    async def _save_evaluation(self, image_path: str, evaluation_text: str, 
                             needs_refinement: bool = False):
        """Save evaluation result to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO image_evaluations 
                (image_path, evaluation_text, needs_refinement)
                VALUES (?, ?, ?)
            """, (str(image_path), evaluation_text, needs_refinement))

    async def evaluate_image(self, image: Image.Image, 
                           image_path: Optional[str] = None) -> Optional[Dict]:
        """Evaluate image using Gemini API and store results."""
        try:
            if not isinstance(image, Image.Image):
                raise ValueError("Input must be a PIL Image object")

            # Generate evaluation using Gemini
            response = self.model.generate_content([
                "Describe this image in detail. Focus on the visual elements present, "
                "such as objects, colors, lighting, composition, and overall scene. "
                "Also indicate if the image needs refinement based on quality and "
                "adherence to typical artistic standards.",
                image
            ])
            response.resolve()

            if not response.parts:
                logger.warning("No description received from Gemini API")
                return None

            evaluation_text = response.text
            needs_refinement = "needs refinement" in evaluation_text.lower()

            # Save evaluation if image path provided
            if image_path:
                await self._save_evaluation(
                    image_path, evaluation_text, needs_refinement
                )

            logger.info(f"Evaluation completed{' - needs refinement' if needs_refinement else ''}")
            return {
                "evaluation_text": evaluation_text,
                "needs_refinement": needs_refinement,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            return None

    async def get_evaluation_history(self, image_path: str) -> Optional[Dict]:
        """Retrieve evaluation history for an image."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT evaluation_text, created_at, needs_refinement
                    FROM image_evaluations
                    WHERE image_path = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (str(image_path),))
                result = cursor.fetchone()
                
                return {
                    "evaluation_text": result[0],
                    "timestamp": result[1],
                    "needs_refinement": bool(result[2])
                } if result else None

        except Exception as e:
            logger.error(f"Error retrieving evaluation history: {str(e)}")
            return None

    async def setup(self):
        """Placeholder for setup - maintained for interface consistency."""
        pass

    async def cleanup(self):
        """Placeholder for cleanup - maintained for interface consistency."""
        pass

    async def __aenter__(self):
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
