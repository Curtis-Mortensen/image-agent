import google.generativeai as genai
import logging
from typing import Optional, Dict
import sqlite3
from pathlib import Path
from datetime import datetime
from config import DATABASE_PATH

from src.prompt_handler import PromptHandler

logger = logging.getLogger(__name__)

class PromptRefiner:
    """Handles prompt refinement using Google Gemini API with database tracking."""

    def __init__(self, api_key: str, data_base_path: Path, prompt_handler: PromptHandler,
                 db_path: str = DATABASE_PATH):
        self.api_key = api_key
        self.data_base_path = data_base_path
        self.prompt_handler = prompt_handler
        self.db_path = db_path
        self.adherence_phrase = "Image adheres to the prompt"
        
        # Initialize Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize database table for refined prompts."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS refined_prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_prompt_id TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    refined_prompt TEXT NOT NULL,
                    evaluation_text TEXT NOT NULL,
                    needs_refinement BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (original_prompt_id) REFERENCES prompts(id),
                    UNIQUE(original_prompt_id, iteration)
                )
            """)

    async def _save_refined_prompt(self, prompt_id: str, original_prompt: str,
                                 evaluation_text: str, refined_content: str,
                                 needs_refinement: bool) -> None:
        """Save refined prompt to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get current max iteration for this prompt
                cursor = conn.execute("""
                    SELECT COALESCE(MAX(iteration), 0) + 1
                    FROM refined_prompts
                    WHERE original_prompt_id = ?
                """, (prompt_id,))
                next_iteration = cursor.fetchone()[0]

                # Save refined prompt
                conn.execute("""
                    INSERT INTO refined_prompts
                    (original_prompt_id, iteration, refined_prompt, 
                     evaluation_text, needs_refinement)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    prompt_id,
                    next_iteration,
                    refined_content,
                    evaluation_text,
                    needs_refinement
                ))

                logger.info(f"Saved refined prompt for {prompt_id}, iteration {next_iteration}")

        except Exception as e:
            logger.error(f"Error saving refined prompt: {str(e)}")

    async def get_refinement_history(self, prompt_id: str) -> list:
        """Retrieve refinement history for a prompt."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT iteration, refined_prompt, evaluation_text,
                           needs_refinement, created_at
                    FROM refined_prompts
                    WHERE original_prompt_id = ?
                    ORDER BY iteration DESC
                """, (prompt_id,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error retrieving refinement history: {str(e)}")
            return []

    async def refine_prompt(self, original_prompt: str, prompt_id: str,
                           evaluation_result: Dict) -> Optional[str]:
        """Refine a prompt using Gemini API and store results."""
        try:
            if not evaluation_result or 'evaluation_text' not in evaluation_result:
                logger.warning("No evaluation result provided")
                return None

            evaluation_text = evaluation_result['evaluation_text']
            needs_refinement = True

            if self.adherence_phrase.lower() in evaluation_text.lower():
                logger.info("Image adheres to prompt, no refinement needed")
                needs_refinement = False
                return original_prompt

            prompt_instruction = (
                "You are evaluating if a generated image adheres to a user's prompt. "
                "Focus on identifying significant deviations:\n"
                "- Missing major elements explicitly requested\n"
                "- Unexpected prominent elements\n"
                "- Large differences in key elements\n"
                f"If no significant deviations, respond with: '{self.adherence_phrase}'\n"
                "Otherwise, suggest a revised prompt.\n\n"
                f"Original Prompt: {original_prompt}\n"
                f"Image Description: {evaluation_text}\n"
                "Analysis and Refinement:"
            )

            response = self.model.generate_content(prompt_instruction)
            response.resolve()

            if not response.text:
                logger.warning("No response from Gemini API")
                return original_prompt

            refined_content = response.text
            await self._save_refined_prompt(
                prompt_id,
                original_prompt,
                evaluation_text,
                refined_content,
                needs_refinement
            )

            return refined_content

        except Exception as e:
            logger.error(f"Error in prompt refinement: {str(e)}")
            return original_prompt

    async def setup(self):
        """Ensure required directories exist."""
        (self.data_base_path / "refined_prompts").mkdir(parents=True, exist_ok=True)

    async def cleanup(self):
        """Cleanup resources."""
        pass

    async def __aenter__(self):
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
