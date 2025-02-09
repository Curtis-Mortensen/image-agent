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

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment:
```bash
cp .env.local.example .env.local
# Edit .env.local with your API keys
```

5. Set up your prompts:
```bash
cp prompts.json.example .prompts.json
# Edit .prompts.json with your prompts
```

## Usage

1. Ensure your prompts are properly formatted in `.prompts.json`:
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

## Monitoring Progress

- Real-time console output shows generation progress
- Check the log file (generation_log_[timestamp].log)
- Review summary.json files in each prompt's output directory

## Configuration

Adjust settings in `config.py`:
- Image generation parameters
- Batch processing size
- API configurations
- Output paths and formats

## Advanced Usage

### Customizing Generation Parameters

Modify the default parameters in `config.py`:
```python
IMAGE_SIZE = (1024, 1024)
BATCH_SIZE = 3
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
```

### API Rate Limiting

The system handles API rate limits automatically with exponential backoff:
- FAL.ai API: Retries with configurable delays
- Gemini API: Built-in rate limit handling

## Troubleshooting

Common issues:

1. **API Authentication**:
   - Verify API keys in .env.local
   - Check API key permissions

2. **Dependencies**:
   - Ensure all requirements are installed: `pip install -r requirements.txt`
   - Check Python version (3.7+ required)

3. **Input/Output**:
   - Verify .prompts.json format
   - Check write permissions for output directory

4. **Rate Limits**:
   - Monitor API usage
   - Adjust batch size if needed

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
```bash
black src/
flake8 src/
```

## License

[Your License Here]

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
