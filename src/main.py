"""
Main entry point for the image generation pipeline.
"""

import asyncio
import logging
import sys
from pathlib import Path
import signal
from datetime import datetime
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.logging import RichHandler
import json

from src.PipelineOrchestrator import PipelineOrchestrator
from src.PromptGenerator import PromptGenerator
from config import (
    FAL_KEY, GEMINI_API_KEY, INPUT_FILE_PATH, 
    OUTPUT_BASE_PATH, PIPELINE_CONFIG
)

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

def create_progress() -> Progress:
    """Create a progress bar instance."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    )

async def generate_prompts(pipeline: PipelineOrchestrator):
    """Generate prompts for scenes that don't have them."""
    try:
        logger.info("Starting prompt generation process...")
        generator = PromptGenerator(GEMINI_API_KEY)
        
        with create_progress() as progress:
            task = progress.add_task("Processing prompts", total=100)
            
            def update_progress(message: str):
                progress.update(task, description=message)
            
            # Generate prompts and update JSON file
            stats = await generator.update_json_file(
                INPUT_FILE_PATH, 
                progress_callback=update_progress
            )
            
            # Show generation statistics
            console.print("\n[bold]Prompt Generation Results:[/bold]")
            console.print(f"Total scenes: {stats['total_scenes']}")
            console.print(f"New prompts generated: {stats['scenes_generated']}")
            console.print(f"Existing prompts: {stats['scenes_existing']}")
            
            progress.update(task, advance=100, description="Complete")
            
    except Exception as e:
        logger.error(f"Error generating prompts: {str(e)}", exc_info=True)
        console.print(f"\n[red]Error generating prompts: {str(e)}[/red]")

async def generate_batch(pipeline: PipelineOrchestrator):
    """Generate a batch of image variants."""
    try:
        prompt_id = input("Enter prompt ID: ")
        with open(INPUT_FILE_PATH) as f:
            prompts = json.load(f)["prompts"]
            prompt_data = next((p for p in prompts if p["id"] == prompt_id), None)
        
        if not prompt_data:
            console.print(f"\n[red]No prompt found with ID: {prompt_id}[/red]")
            return
            
        batch_size = input("Enter batch size (3-7, default 5): ")
        batch_size = int(batch_size) if batch_size.isdigit() else None
        
        with create_progress() as progress:
            task = progress.add_task("Generating batch", total=1)
            
            variants = await pipeline.generate_batch_variants(
                prompt_id, prompt_data, 1, batch_size, task
            )
            
            if variants:
                console.print("\n[green]Generated Variants:[/green]")
                for v in variants:
                    console.print(f"- {v['image_path']}")
            
            progress.update(task, advance=1)
            
    except Exception as e:
        logger.error(f"Error generating batch: {str(e)}")
        console.print(f"\n[red]Error: {str(e)}[/red]")

async def evaluate_and_select(pipeline: PipelineOrchestrator):
    """Evaluate images and select the best variant."""
    try:
        # Get image paths
        image_dir = input("Enter directory containing images: ")
        image_dir = Path(image_dir)
        if not image_dir.exists():
            console.print(f"\n[red]Directory not found: {image_dir}[/red]")
            return
            
        # Get prompt
        prompt = input("Enter prompt to evaluate against: ")
        
        # Create variants list from images
        variants = [
            {
                'image_path': str(p),
                'variant': i
            } for i, p in enumerate(image_dir.glob("*.png"))
        ]
        
        if not variants:
            console.print("\n[red]No images found in directory[/red]")
            return
            
        with create_progress() as progress:
            # Evaluate variants
            task = progress.add_task("Processing images", total=2)
            
            evaluated = await pipeline.evaluate_variants(variants, prompt, task)
            if evaluated:
                console.print("\n[green]Evaluation Results:[/green]")
                for v in evaluated:
                    console.print(f"\nImage: {v['image_path']}")
                    console.print(f"Score: {v['evaluation_score']:.2f}")
                    console.print(f"Feedback: {v['feedback']}")
            
            progress.update(task, advance=1)
            
            # Select best
            best = await pipeline.select_best_variant(evaluated, "manual", 1)
            if best:
                console.print("\n[green]Best Variant:[/green]")
                console.print(f"Image: {best['image_path']}")
                console.print(f"Score: {best['evaluation_score']:.2f}")
            
            progress.update(task, advance=1)
            
    except Exception as e:
        logger.error(f"Error evaluating images: {str(e)}")
        console.print(f"\n[red]Error: {str(e)}[/red]")

async def evaluate_single_image(pipeline: PipelineOrchestrator, image_path: Path):
    """Evaluate a single image on demand."""
    try:
        with create_progress() as progress:
            task = progress.add_task("Evaluating image", total=1)
            
            result = await pipeline.evaluate_single_image(image_path)
            if result:
                console.print("\n[green]Image Description:[/green]")
                console.print(result['description'])
                console.print(f"\nDescription saved to: {result['description_path']}")
            
            progress.update(task, advance=1)
            
    except Exception as e:
        logger.error(f"Error evaluating image: {str(e)}")
        console.print(f"\n[red]Error: {str(e)}[/red]")

async def refine_single_prompt(pipeline: PipelineOrchestrator, 
                             prompt: str, description: str):
    """Refine a single prompt on demand."""
    try:
        with create_progress() as progress:
            task = progress.add_task("Refining prompt", total=1)
            
            refined = await pipeline.refine_single_prompt(prompt, description)
            if refined:
                console.print("\n[green]Refined Prompt:[/green]")
                console.print(refined)
            else:
                console.print("\n[yellow]No refinement needed[/yellow]")
            
            progress.update(task, advance=1)
            
    except Exception as e:
        logger.error(f"Error refining prompt: {str(e)}")
        console.print(f"\n[red]Error: {str(e)}[/red]")

async def main_menu():
    """Interactive menu interface."""
    try:
        logger.info("Initializing pipeline...")
        progress = create_progress()
        
        pipeline = PipelineOrchestrator(
            INPUT_FILE_PATH,
            OUTPUT_BASE_PATH,
            FAL_KEY,
            GEMINI_API_KEY,
            progress
        )
        
        logger.info(f"Pipeline initialized with:"
                   f"\n  - FAL_KEY={FAL_KEY[:8]}..."
                   f"\n  - GEMINI_API_KEY={GEMINI_API_KEY[:8]}..."
                   f"\n  - Max iterations: {PIPELINE_CONFIG['max_iterations']}"
                   f"\n  - Batch size: {PIPELINE_CONFIG['batch_size']}")

        menu_options = {
            "1": ("Generate Missing Prompts", 
                lambda: generate_prompts(pipeline)),
            "2": ("Generate Image Batch", 
                lambda: generate_batch(pipeline)),
            "3": ("Evaluate and Select Best Images", 
                lambda: evaluate_and_select(pipeline)),
            "4": ("Full Pipeline (Multiple Iterations)", 
                lambda: pipeline.run_pipeline(batch_only=False)),
            "5": ("Evaluate Single Image", 
                lambda: evaluate_single_image(pipeline, 
                    Path(input("Enter image path: ")))),
            "6": ("Refine Single Prompt", 
                lambda: refine_single_prompt(pipeline,
                    input("Enter prompt: "),
                    input("Enter image description: "))),
        }

        while True:
            console.print("\n[bold]Choose an action:[/bold]")
            for key, (desc, _) in menu_options.items():
                console.print(f"{key}. {desc}")
            console.print("7. Exit")

            choice = input("\nChoice (1-7): ")
            if choice == "7":
                break
            elif choice in menu_options:
                try:
                    await menu_options[choice][1]()
                except Exception as e:
                    logger.error(f"Error executing option {choice}: {str(e)}", 
                               exc_info=True)
                    console.print(f"\n[red]Error: {str(e)}[/red]")
            else:
                console.print("[red]Invalid choice[/red]")
                
    except Exception as e:
        logger.error(f"Error in main menu: {str(e)}", exc_info=True)
        console.print(f"\n[red]Error initializing: {str(e)}[/red]")

@click.command()
def main_cli():
    """CLI entry point."""
    asyncio.run(main_menu())

if __name__ == '__main__':
    main_cli()
