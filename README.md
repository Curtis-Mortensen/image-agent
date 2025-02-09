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

## Installation

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

## Usage

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
python -m src.main
```

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

## Configuration

Adjust settings in `config.py`:
- Image size
- Batch size
- API timeouts
- Retry settings
- Output paths

## License

[Your License Here]

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
