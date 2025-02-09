# AI Image Generation Batch Processor Project Description

This project is a Python-based batch processing system for AI image generation that automates the creation and refinement of AI-generated images. Here's the core concept:

## Project Purpose
The system takes a collection of image prompts (with context like scene descriptions, moods, and titles) and automatically:
1. Generates images using the fal.ai API
2. Evaluates the results using Google's Gemini Vision API
3. Creates a refined prompt based on the evaluation
4. Generates a second iteration of the image
5. Saves all results, including original prompts, generated images, evaluations, and refined prompts

## Key Features
- Reads prompts and context from a JSON file
- Interfaces with fal.ai's image generation APIs
- Uses Gemini Vision API for image evaluation
- Generates refined prompts using Gemini
- Saves all iterations and metadata in an organized directory structure
- Handles errors and API rate limits gracefully

## Input Format
The system accepts JSON files containing prompts structured like this:
```json
{
    "prompts": [
        {
            "id": "scene_001",
            "title": "Forest Awakening",
            "scene": "Deep in a mystical forest",
            "mood": "Ethereal, peaceful",
            "prompt": "A shaft of golden morning light piercing through misty ancient trees"
        }
    ]
}
```

## Output Structure
For each prompt, the system creates a directory containing:
- The original prompt and context
- First generation image
- Gemini's evaluation
- Refined prompt
- Second generation image
- All associated metadata

## Technical Requirements
- Must handle API authentication for both fal.ai and Gemini
- Must implement proper error handling and logging
- Must create organized, human-readable output directories
- Must handle batch processing efficiently
- Must save all iterations and metadata for future reference

This is designed as a minimum viable product (MVP) that can be extended later with features like:
- Multiple model comparison
- Success criteria for continued iterations
- GUI interface
- Real-time progress tracking
- Additional AI model integrations

The system should be built to be modular and extensible, allowing for easy addition of new features and API integrations in the future.

Here's a detailed description for each Python file in the project:

**config.py**
- Store API keys and configuration settings
- Constants should include:
  - FAL_AI_API_KEY
  - GEMINI_API_KEY
  - OUTPUT_BASE_PATH (for storing generated images)
  - INPUT_FILE_PATH (path to prompts.json)
  - MODEL_ID (e.g., "sd-1.5", "sdxl", etc.)
  - BATCH_SIZE (how many prompts to process at once)
  - IMAGE_SIZE (tuple of width, height)

**src/main.py**
- Main entry point that orchestrates the entire process
- Should import PromptHandler and ImageGenerator classes
- Create a main() function that:
  1. Loads prompts from json using PromptHandler
  2. For each prompt:
     - Generates image using ImageGenerator
     - Gets evaluation from Gemini
     - Creates refined prompt
     - Generates second image
     - Saves all results
- Include basic logging setup
- Include if __name__ == "__main__" block

**src/prompt_handler.py**
- Create PromptHandler class
- Methods needed:
  - load_prompts(): reads json file
  - save_results(prompt_id, iteration, image_path, evaluation, refined_prompt)
  - create_output_directory(prompt_id)
  - get_refined_prompt(original_prompt, evaluation): uses Gemini to create new prompt
- Should handle all file I/O operations
- Should create necessary directories in outputs/generations/

**src/image_generator.py**
- Create ImageGenerator class
- Methods needed:
  - generate_image(prompt, prompt_id, iteration): calls fal.ai API
  - evaluate_image(image_path): calls Gemini Vision API
- Should handle all API interactions
- Should include error handling for API failures
- Should save images to appropriate directories using PromptHandler

**src/api_client.py**
- Create separate clients for each API service
- FalClient class:
  - initialize with API key
  - method for sending requests to fal.ai
  - handle response parsing
- GeminiClient class:
  - initialize with API key
  - method for sending image for evaluation
  - method for generating refined prompts
- Include retry logic and error handling

Each file should use type hints and include proper documentation. The system should be designed to handle API rate limits and potential failures gracefully. All sensitive information should be loaded from environment variables or a secure configuration file.

The flow should be:
1. Load prompts from JSON
2. For each prompt:
   - Create output directory structure
   - Generate first image
   - Save image and metadata
   - Evaluate with Gemini
   - Generate refined prompt
   - Generate second image
   - Save all results and metadata

Error handling should include:
- API failures
- File I/O errors
- Invalid prompts
- Network issues
- Rate limiting

Logging should track:
- Start/end of each generation
- API calls
- Errors
- Timing information
