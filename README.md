# AI Image Generation Pipeline

A Python-based system for batch processing and refinement of AI-generated images using FAL.ai and Google's Gemini API.

## Features

- Batch image generation with FAL.ai
- Automated image evaluation using Gemini Vision
- Prompt refinement based on evaluations
- Concurrent processing for efficiency
- Comprehensive result tracking and organization

## Prerequisites

- Python 3.7+
- FAL.ai API key
- Google Gemini API key

## Important Note About Dependencies

This project requires several external packages that need to be installed in a full Python environment:
```
aiohttp>=3.8.0
Pillow>=10.0.0
google-generativeai>=0.3.0
python-dotenv>=1.0.0
asyncio>=3.4.3
pathlib>=1.0.1
```

However, since we're running in WebContainer which only supports Python standard library, we've adapted the code to work within these limitations.

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd image-agent
```

2. Set up environment variables:
```bash
export FAL_AI_API_KEY="your_fal_ai_key_here"
export GEMINI_API_KEY="your_gemini_key_here"
```

## Running the Program

1. Place your prompts in `data/inputs/prompts.json`:
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

2. Run the pipeline:
```bash
python3 src/main.py
```

The program will:
1. Load prompts from your JSON file
2. Process each prompt in batches
3. Generate initial images
4. Evaluate the results
5. Generate refined versions
6. Save all outputs in the results directory

## Output Structure

```
outputs/
└── results/
    └── prompt_id/
        ├── iteration_1.png
        ├── iteration_1_results.json
        ├── iteration_1_evaluation.json
        ├── iteration_2.png
        ├── iteration_2_results.json
        └── summary.json
```

## Monitoring Progress

- Check the console output for real-time progress
- Review the generated log file in the project directory
- Examine the summary.json files in each prompt's output directory

## Configuration

Adjust settings in `config.py`:
- Image size
- Batch size
- API timeouts
- Retry settings
- Output paths

## Troubleshooting

Common issues:
1. **Environment Variables**: Make sure both API keys are properly set
2. **Input JSON**: Verify your prompts.json follows the required format
3. **Permissions**: Ensure the program has write access to create output directories
4. **API Limits**: Check if you've hit any API rate limits

## License

[Your License Here]

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
# AI Image Generation Pipeline

A Python-based system for batch processing and refinement of AI-generated images using FAL.ai and Google's Gemini API.

## Features

- Batch image generation with FAL.ai
- Automated image evaluation using Gemini Vision
- Prompt refinement based on evaluations
- Concurrent processing for efficiency
- Comprehensive result tracking and organization

## Prerequisites

- Python 3.7+
- FAL.ai API key
- Google Gemini API key

## Important Note About Dependencies

This project requires several external packages that need to be installed in a full Python environment:
```
aiohttp>=3.8.0
Pillow>=10.0.0
google-generativeai>=0.3.0
python-dotenv>=1.0.0
asyncio>=3.4.3
pathlib>=1.0.1
```

However, since we're running in WebContainer which only supports Python standard library, we've adapted the code to work within these limitations.

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd image-agent
```

2. Set up environment variables:
```bash
export FAL_AI_API_KEY="your_fal_ai_key_here"
export GEMINI_API_KEY="your_gemini_key_here"
```

## Running the Program

1. Place your prompts in `data/inputs/prompts.json`:
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

2. Run the pipeline:
```bash
python3 src/main.py
```

The program will:
1. Load prompts from your JSON file
2. Process each prompt in batches
3. Generate initial images
4. Evaluate the results
5. Generate refined versions
6. Save all outputs in the results directory

## Output Structure

```
outputs/
└── results/
    └── prompt_id/
        ├── iteration_1.png
        ├── iteration_1_results.json
        ├── iteration_1_evaluation.json
        ├── iteration_2.png
        ├── iteration_2_results.json
        └── summary.json
```

## Monitoring Progress

- Check the console output for real-time progress
- Review the generated log file in the project directory
- Examine the summary.json files in each prompt's output directory

## Configuration

Adjust settings in `config.py`:
- Image size
- Batch size
- API timeouts
- Retry settings
- Output paths

## Troubleshooting

Common issues:
1. **Environment Variables**: Make sure both API keys are properly set
2. **Input JSON**: Verify your prompts.json follows the required format
3. **Permissions**: Ensure the program has write access to create output directories
4. **API Limits**: Check if you've hit any API rate limits

## License

[Your License Here]

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
