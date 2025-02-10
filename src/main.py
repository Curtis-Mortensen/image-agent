import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List
import signal
import sqlite3
import click
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.logging import RichHandler
from PIL import Image
import aiofiles
from concurrent.futures import ProcessPoolExecutor

from src.image_generator import ImageGenerator
from src.prompt_handler import PromptHandler
from src.image_evaluator import ImageEvaluator
from src.prompt_refiner import PromptRefiner
from config import FAL_KEY, GEMINI_API_KEY, INPUT_FILE_PATH, OUTPUT_BASE_PATH

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "image_generation.db"):
        self.db_path = db_path
        self.conn = None
        self._create_tables()
        self._initialize_version()

    def _initialize_version(self):
        """Initialize version info if not already present."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM version_info")
            if cursor.fetchone()[0] == 0:
                conn.execute("""
                    INSERT INTO version_info (version)
                    VALUES ('1.0.0')
                """)
                conn.commit()

    def _create_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Core tables
                CREATE TABLE IF NOT EXISTS scenes (
                    id TEXT PRIMARY KEY,
                    original_prompt TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    current_iteration INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS iterations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scene_id TEXT,
                    version INTEGER,
                    prompt TEXT NOT NULL,
                    image_path TEXT,
                    evaluation_text TEXT,
                    needs_refinement BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (scene_id) REFERENCES scenes(id)
                );

                CREATE TABLE IF NOT EXISTS status (
                    scene_id TEXT PRIMARY KEY,
                    current_version INTEGER,
                    is_complete BOOLEAN DEFAULT FALSE,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (scene_id) REFERENCES scenes(id)
                );

                -- API tracking tables
                CREATE TABLE IF NOT EXISTS api_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    api_name TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Version tracking
                CREATE TABLE IF NOT EXISTS version_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL,
                    installed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

    async def save_iteration(self, scene_id: str, version: int, prompt: str, 
                           image_path: str, evaluation: Optional[Dict] = None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO iterations (scene_id, version, prompt, image_path, evaluation_text)
                VALUES (?, ?, ?, ?, ?)
            """, (scene_id, version, prompt, str(image_path), 
                  evaluation['evaluation_text'] if evaluation else None))
            conn.execute("""
                UPDATE scenes SET current_iteration = ?
                WHERE id = ?
            """, (version, scene_id))

    async def get_scene_status(self, scene_id: str) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT current_iteration, status FROM scenes WHERE id = ?
            """, (scene_id,))
            return dict(cursor.fetchone())

class ImageGenerationPipeline:
    def __init__(self, input_file_path: Path, output_base_path: Path,
             fal_api_key: str, gemini_api_key: str):
        self.db = DatabaseManager(str(output_base_path / "image_generation.db"))
        self.prompt_handler = PromptHandler(input_file_path, output_base_path)
        self.image_generator = ImageGenerator(fal_api_key, output_base_path)
        self.image_evaluator = ImageEvaluator(gemini_api_key)
        self.prompt_refiner = PromptRefiner(gemini_api_key, output_base_path, self.prompt_handler)
        self.running = True
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        )
        self.process_pool = ProcessPoolExecutor(max_workers=2)

    def setup_signal_handlers(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}. Starting graceful shutdown...")
        self.running = False
        self.process_pool.shutdown(wait=False)
    
    async def process_iteration(self, prompt_id: str, prompt_data: dict, 
                              iteration: int, progress_task: TaskID) -> bool:
        """Combined generation, evaluation, and refinement for one iteration."""
        try:
            # Generate image
            image_path = await self.image_generator.process_prompt(
                prompt_id, prompt_data, iteration,
                lambda msg: self.progress.update(progress_task, description=msg)
            )
            if not image_path:
                return False

            # Evaluate image
            evaluation = await self.image_evaluator.evaluate_image(Image.open(image_path))
            if not evaluation:
                return False

            # Save iteration data
            await self.db.save_iteration(
                prompt_id, iteration, prompt_data['prompt'],
                str(image_path), evaluation
            )

            # Refine prompt if needed and not final iteration
            if iteration < 3 and evaluation.get('needs_refinement'):
                refined_prompt = await self.prompt_refiner.refine_prompt(
                    prompt_data['prompt'], prompt_id, evaluation
                )
                if refined_prompt:
                    prompt_data['prompt'] = refined_prompt

            return True

        except Exception as e:
            logger.error(f"Error in iteration {iteration} for {prompt_id}: {str(e)}")
            return False

    async def run_pipeline(self, batch_only: bool = False) -> None:
        """Unified pipeline runner for both batch and full processing."""
        try:
            async with self.prompt_handler, self.image_generator:
                prompts = await self.prompt_handler.load_prompts()
                if not prompts:
                    logger.error("No prompts found.")
                    return

                max_iterations = 1 if batch_only else 3
                logger.info(f"Starting pipeline for {len(prompts)} prompts, "
                          f"max {max_iterations} iterations{'(batch mode)' if batch_only else ''}")

                await self.image_evaluator.setup()
                await self.prompt_refiner.setup()

                for prompt_id, prompt_data in prompts.items():
                    if not self.running:
                        break

                    task = self.progress.add_task(
                        f"Processing {prompt_id}", total=max_iterations
                    )
                    
                    for iteration in range(1, max_iterations + 1):
                        if not await self.process_iteration(
                            prompt_id, prompt_data, iteration, task
                        ):
                            break
                        self.progress.update(task, advance=1)

                logger.info("Pipeline completed successfully.")

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
        finally:
            await self.image_evaluator.cleanup()
            await self.prompt_refiner.cleanup()
            self.process_pool.shutdown(wait=True)

async def evaluate_single_image(pipeline: ImageGenerationPipeline, image_path: Path):
    """Evaluate a single image on demand."""
    try:
        await pipeline.image_evaluator.setup()
        task = pipeline.progress.add_task("Evaluating image", total=1)
        
        evaluation = await pipeline.image_evaluator.evaluate_image(Image.open(image_path))
        if evaluation:
            console.print("\n[green]Evaluation Result:[/green]")
            console.print(evaluation['evaluation_text'])
            
            eval_path = image_path.parent / f"{image_path.stem}_evaluation.txt"
            async with aiofiles.open(eval_path, 'w') as f:
                await f.write(evaluation['evaluation_text'])
            console.print(f"\nEvaluation saved to: {eval_path}")
        
        pipeline.progress.update(task, advance=1)
    finally:
        await pipeline.image_evaluator.cleanup()

async def refine_single_prompt(pipeline: ImageGenerationPipeline, 
                             prompt: str, evaluation: str):
    """Refine a single prompt on demand."""
    try:
        await pipeline.prompt_refiner.setup()
        task = pipeline.progress.add_task("Refining prompt", total=1)
        
        refined = await pipeline.prompt_refiner.refine_prompt(
            prompt, "manual_prompt", {'evaluation_text': evaluation}
        )
        if refined:
            console.print("\n[green]Refined Prompt:[/green]")
            console.print(refined)
        
        pipeline.progress.update(task, advance=1)
    finally:
        await pipeline.prompt_refiner.cleanup()

async def main_menu():
    """Simplified menu interface."""
    pipeline = ImageGenerationPipeline(
        INPUT_FILE_PATH,
        OUTPUT_BASE_PATH,
        FAL_KEY,
        GEMINI_API_KEY
    )

    menu_options = {
        "1": ("Batch Generation (First Iteration)", 
              lambda: pipeline.run_pipeline(batch_only=True)),
        "2": ("Full Pipeline (Up to 3 Iterations)", 
              lambda: pipeline.run_pipeline(batch_only=False)),
        "3": ("Evaluate Single Image", 
              lambda: evaluate_single_image(pipeline, 
                  Path(input("Enter image path: ")))),
        "4": ("Refine Single Prompt", 
              lambda: refine_single_prompt(pipeline,
                  input("Enter prompt: "),
                  input("Enter evaluation: "))),
    }

    while True:
        console.print("\n[bold]Choose an action:[/bold]")
        for key, (desc, _) in menu_options.items():
            console.print(f"{key}. {desc}")
        console.print("5. Exit")

        choice = input("\nChoice (1-5): ")
        if choice == "5":
            break
        elif choice in menu_options:
            await menu_options[choice][1]()
        else:
            console.print("[red]Invalid choice[/red]")

@click.command()
def main_cli():
    """CLI entry point."""
    asyncio.run(main_menu())

if __name__ == '__main__':
    main_cli()
