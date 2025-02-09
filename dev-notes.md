# AI Image Generation Batch Processor Project Description

This project is a Python-based batch processing system for AI image generation that automates the creation and refinement of AI-generated images. Here's the core concept:

## Project Purpose
The system takes a collection of image prompts (with context like scene descriptions, moods, and titles) and automatically:
1. Generates images using the fal.ai API
2. Evaluates the results using Google's Gemini Vision API
3. Creates a refined prompt based on the evaluation
4. Generates a second iteration of the image
5. Saves all results, including original prompts, generated images, evaluations, and refined prompts

## Architecture & Standards

### Async/Sync Standards
- All API interactions (fal.ai, Gemini) are async
- File I/O operations remain synchronous (no async benefit)
- Main execution flow is async for concurrent processing
- Batch processing uses asyncio.gather for parallel execution
- Signal handlers are synchronous (OS requirement)

### Class Structure
1. **ImageGenerationPipeline** (main.py)
   - Orchestrates overall process
   - Handles batch processing
   - Manages graceful shutdown
   - Coordinates between components

2. **ImageGenerator** (image_generator.py)
   - Handles image generation logic
   - Manages API client interactions
   - Coordinates evaluation and refinement
   - Async methods for API operations

3. **PromptHandler** (prompt_handler.py)
   - Manages prompt loading and validation
   - Handles result saving
   - Synchronous file operations
   - Maintains output organization

4. **API Clients** (api_client.py)
   - FalClient: Handles fal.ai API interactions
   - GeminiClient: Manages Gemini API operations
   - Async methods for API calls
   - Handles rate limiting and retries

### Error Handling Standards
1. **API Errors**
   - Retry logic for transient failures
   - Rate limit handling with exponential backoff
   - Detailed error logging
   - Graceful degradation

2. **File Operations**
   - Path existence checks
   - Permission validation
   - Atomic write operations where possible
   - Detailed error messages

3. **Data Validation**
   - Input prompt validation
   - Data structure verification
   - Type checking
   - Required field validation

### Logging Standards
- Hierarchical logger setup
- Both file and console logging
- Structured log format
- Different log levels for different purposes
- Timestamp in log filename

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

## Key Implementation Details

### Batch Processing
- Default batch size: 3 concurrent operations
- Configurable through ImageGenerator.BATCH_SIZE
- Uses asyncio.gather for parallel execution
- Maintains task list for cleanup

### Resource Management
- Proper async context management
- Cleanup in finally blocks
- Signal handler for graceful shutdown
- Task cancellation handling

### API Integration
1. **FAL.ai Integration**
   - Async HTTP requests
   - Rate limit handling
   - Retry mechanism
   - Error handling

2. **Gemini Integration**
   - Vision API for evaluation
   - Text API for prompt refinement
   - Async operation handling
   - Response parsing

### File Management
- Organized directory structure
- Atomic write operations
- JSON for metadata storage
- Summary file maintenance

### Performance Considerations
- Batch processing for optimal throughput
- Concurrent API calls
- Efficient resource cleanup
- Memory management

## Technical Requirements
- Python 3.7+
- Async/await support
- JSON processing
- Image handling
- HTTP client support

## Dependencies
```python
aiohttp        # Async HTTP client
Pillow         # Image processing
google.generativeai  # Gemini API
```

## Error Handling Strategy
1. **API Errors**
   - Retry with exponential backoff
   - Rate limit handling
   - Error logging
   - Graceful degradation

2. **File Operations**
   - Path validation
   - Permission checks
   - Atomic operations
   - Error reporting

3. **Data Validation**
   - Schema validation
   - Type checking
   - Required fields
   - Format verification

## Logging Strategy
1. **Levels**
   - ERROR: Operation failures
   - WARNING: Non-critical issues
   - INFO: Operation progress
   - DEBUG: Detailed information

2. **Format**
   ```
   %(asctime)s - %(name)s - %(levelname)s - %(message)s
   ```

3. **Output**
   - Console output
   - File logging
   - Timestamp in filename

## Best Practices
1. **Code Organization**
   - Clear class responsibilities
   - Proper error handling
   - Comprehensive logging
   - Type hints

2. **Resource Management**
   - Proper cleanup
   - Signal handling
   - Task management
   - Memory efficiency

3. **API Integration**
   - Rate limit respect
   - Retry logic
   - Error handling
   - Response validation

4. **File Operations**
   - Safe write operations
   - Directory management
   - Path handling
   - Error checking
