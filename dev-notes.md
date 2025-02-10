# AI Image Generation Pipeline - Development Notes

## Architecture Overview

### Core Components

# Program Purpose and Pipeline Description

This program is designed as a modular system for AI-assisted image generation, evaluation, and prompt refinement. It is structured into distinct modules to handle each stage of the image creation process, allowing for flexibility and maintainability.

## Modular Program Components

1.  **Image Generator**:
    -   **Purpose**: This module is responsible for generating images based on text prompts using the FAL.ai API. It takes a prompt as input and outputs an image file saved to the project's output directory (`data/outputs/images`).
    -   **Modularity**: It is designed to be independent and focused solely on image generation and saving. It does not handle image evaluation or prompt refinement.
    -   **Input**: Text prompts (including title, scene, mood, and the main prompt).
    -   **Output**: Image files (PNG format) saved in the `data/outputs/images` directory.

2.  **Image Evaluator**:
    -   **Purpose**: This module evaluates images using the Google Gemini API. It takes an image file (already present in the project directory - no import code needed) as input and provides a textual description of the image's content.
    -   **Modularity**: It operates independently, taking an image file path and returning an evaluation. It is not involved in image generation or prompt refinement directly.
    -   **Input**: Path to an image file within the project directory.
    -   **Output**: Textual evaluation of the image content from the Gemini API.

3.  **Prompt Refiner**:
    -   **Purpose**: This module refines text prompts using the Google Gemini API, based on an original prompt and an evaluation of a generated image. It aims to improve the prompt to better achieve the desired image characteristics.
    -   **Modularity**: It works independently, taking an original prompt and an image evaluation as input and suggesting a refined prompt.
    -   **Input**: Original text prompt and textual evaluation of a generated image.
    -   **Output**: Refined text prompt suggested by the Gemini API.

## Image Generation Pipeline

In addition to the modular components, the program also implements an automated image generation pipeline. This pipeline orchestrates the modular components to iteratively generate, evaluate, and refine images. The pipeline process is as follows:

1.  **Batch Image Generation**:
    -   The pipeline starts by loading a batch of initial prompts from a JSON input file (`prompts.json`).
    -   For each prompt in the batch, the `Image Generator` module is used to generate an initial image.
    -   The generated images are saved to the output directory.

2.  **Image Evaluation**:
    -   Once the initial batch of images is generated, the `Image Evaluator` module is used to evaluate each generated image.
    -   The evaluation provides a textual description of each image's content.

3.  **Prompt Refinement**:
    -   For each evaluated image, the `Prompt Refiner` module is used to refine the original prompt, based on the image evaluation.
    -   The goal is to adjust the prompt to better align the generated images with the user's intention.

4.  **Iterative Regeneration (Up to 3x)**:
    -   Steps 1-3 (Generate, Evaluate, Refine) are repeated for up to 3 iterations in total.
    -   In each iteration after the first, the `Image Generator` uses the *refined prompts* from the previous iteration to generate a new batch of images.
    -   The newly generated images are again evaluated, and prompts are further refined.
    -   This iterative process allows for progressive improvement of the generated images through automated feedback and refinement loops.

5.  **Output and Summary**:
    -   Throughout the pipeline, all generated images, evaluations, refined prompts, and summary information are saved in an organized output directory structure (`data/outputs/results`).
    -   Summary files track the progress and results of each prompt through the iterations.

This pipeline is designed to automate and enhance the image generation process, leveraging AI feedback to iteratively improve image quality and prompt effectiveness. The modular design ensures that each component can be understood, maintained, and potentially extended or replaced independently.

## New Features and Improvements

### 1. Enhanced Progress Tracking
- Rich progress bars and spinners
- Real-time status updates
- Visual task progression
- Batch completion statistics

### 2. Improved Error Handling
- Rich tracebacks
- Structured logging
- Error recovery mechanisms
- Graceful degradation

### 3. Resource Management
- Process pools for CPU tasks
- Thread pools for I/O
- Async context managers
- Proper cleanup

### 4. Data Validation
- JSON schema validation
- Dataclass-based structures
- Type hints throughout
- Input sanitization

### 5. File Operations
- Async file handling
- Atomic writes
- Multiple format support (JSON/YAML)
- Structured output organization

### 6. CLI Support
- Command-line interface
- Configuration options
- Runtime parameters
- Usage help

## Recent Updates

### FAL.ai API Integration Changes
- Migrated to subscription-based FAL.ai API
- Implemented new `fal_client.subscribe()` approach
- Updated model ID to "fal-ai/fast-lightning-sdxl"
- Added real-time status updates and log streaming
- Improved error handling for API responses

### Architecture Improvements

#### 1. Code Reorganization
- Removed duplicate functionality between main.py and image_generator.py
- Centralized image generation logic in ImageGenerator class
- Simplified pipeline orchestration in main.py
- Improved separation of concerns

#### 2. New ImageGenerator Features
- Added `generate_and_evaluate` method for complete generation flow
- Implemented progress callback system
- Centralized prompt construction
- Enhanced error handling and logging
- Improved API client initialization

#### 3. Pipeline Refinements
- Streamlined batch processing
- Enhanced progress tracking
- Improved error recovery
- Better resource management

## Dependencies

```python
# Core Dependencies
aiohttp>=3.8.0        # Async HTTP
Pillow>=10.0.0        # Image processing
google-generativeai>=0.3.0  # Gemini API
python-dotenv>=1.0.0  # Environment management

# Enhanced Functionality
click>=8.0.0          # CLI support
rich>=10.0.0          # Progress/Console
aiofiles>=0.8.0       # Async files
pyyaml>=6.0.0         # YAML support
jsonschema>=4.0.0     # Schema validation

# Reliability
backoff>=2.2.0        # Retry mechanism
tenacity>=8.0.0       # Retry policies
```

## Configuration

### Environment Variables
- FAL_AI_API_KEY
- GEMINI_API_KEY
- LOG_LEVEL
- RUN_ID

### Runtime Configuration
- Batch size
- Worker pools
- Retry policies
- Output formats

## Output Structure

```
outputs/
└── results/
    └── prompt_id/
        ├── iteration_1.png
        ├── iteration_1_results.json
        ├── iteration_1_results.yaml
        ├── iteration_1_evaluation.json
        ├── iteration_2.png
        ├── iteration_2_results.json
        ├── iteration_2_results.yaml
        └── summary.json
```

## Best Practices

### 1. Async Operations
- Use aiofiles for file operations
- Implement async context managers
- Handle cancellation properly
- Pool connections appropriately

### 2. Resource Management
- Use process pools for CPU tasks
- Use thread pools for I/O
- Implement proper cleanup
- Handle signals gracefully

### 3. Error Handling
- Implement retry mechanisms
- Log structured errors
- Provide rich tracebacks
- Handle edge cases

### 4. Data Management
- Validate input data
- Use structured classes
- Implement atomic writes
- Maintain audit trails

## Usage Examples

### Basic Usage
```bash
python -m src.main
```

### CLI Options
```bash
python -m src.main --input-file custom_prompts.json --output-dir custom_output --batch-size 5
```

### Environment Setup
```bash
cp .env.local.example .env.local
# Edit .env.local with your API keys
```

## Monitoring and Debugging

### Logging
- Console output with rich formatting
- File logging with timestamps
- Structured log levels
- Traceback support

### Progress Tracking
- Visual progress bars
- Task status updates
- Batch statistics
- Error reporting

## Future Improvements

1. **Scalability**
   - Distributed processing
   - Queue system integration
   - Cloud storage support

2. **Monitoring**
   - Metrics collection
   - Performance tracking
   - Resource usage monitoring

3. **Integration**
   - Additional AI models
   - Alternative storage backends
   - Web interface

4. **Features**
   - Image comparison tools
   - Quality metrics
   - Automated optimization


# Development Notes - SQLite Integration and Code Optimization (Date: [Current Date])

## Overview
Today's changes focused on converting the image generation pipeline from file-based storage to SQLite, improving error handling, and optimizing the overall architecture. The changes span across multiple components and introduce new functionality for better data management and reliability.

## Major Changes

### 1. Database Integration
- Implemented SQLite database across all components
- Created centralized database schema for:
  - Prompts and iterations
  - API calls tracking
  - Error logging
  - Version information
  - Status tracking
  - Evaluation results

```sql
-- Core Schema Structure
CREATE TABLE prompts (...)
CREATE TABLE iterations (...)
CREATE TABLE api_calls (...)
CREATE TABLE error_logs (...)
CREATE TABLE version_info (...)
CREATE TABLE refined_prompts (...)
```

### 2. Component Updates

#### ImageEvaluator
- Converted to use SQLite for evaluation storage
- Added evaluation history tracking
- Improved error handling
- Added rate limiting
- Integrated with new database structure

#### PromptHandler
- Replaced JSON/YAML storage with SQLite
- Added prompt versioning
- Improved status tracking
- Enhanced error handling
- Added database-backed caching

#### PromptRefiner
- Integrated with SQLite for refinement tracking
- Added refinement history
- Improved prompt iteration handling
- Enhanced evaluation integration

#### APIClient
- Added API call tracking
- Implemented rate limiting
- Enhanced error recovery
- Added connection pooling
- Improved retry logic

### 3. New Features

#### Database Configuration
- Added DatabaseConfig class
- Implemented connection pooling
- Added transaction management
- Enhanced error handling

#### Configuration Updates
- Reorganized config.py
- Added comprehensive settings for:
  - Database
  - API limits
  - Error recovery
  - Performance
  - Logging

## Technical Details

### Database Connection Management
```python
class DatabaseConfig:
    def __init__(self, db_path: str = "image_generation.db"):
        self.db_path = db_path
        self._connection = None
```

### Key Configuration Settings
```python
DATABASE_CONFIG = {...}
API_LIMITS = {...}
ERROR_RECOVERY = {...}
```

## Future Improvements
1. **Database Optimization**
   - Add indexes for frequent queries
   - Implement query caching
   - Add database migration system

2. **Error Recovery**
   - Add more sophisticated recovery strategies
   - Implement automatic cleanup of old error logs
   - Add recovery analytics

3. **Performance**
   - Implement connection pooling
   - Add query optimization
   - Implement batch processing

4. **Monitoring**
   - Add performance metrics
   - Implement system health checks
   - Add automated alerting

## Dependencies
- SQLite3
- Python 3.8+
- Google Gemini API
- FAL.ai API

## Testing Requirements
- Database operations
- Error recovery scenarios
- API rate limiting
- Connection handling
- Data consistency

## Integration Points
1. Database connections in each component
2. Error recovery system
3. Configuration management
4. API client integration
5. Status tracking system

## Known Issues
- Need to implement database migrations
- Error recovery needs more extensive testing
- Rate limiting needs fine-tuning
- Connection pooling could be optimized

## Migration Notes
When migrating from the previous version:
1. Run database initialization
2. Transfer existing data to SQLite
3. Update configuration settings
4. Test error recovery scenarios
5. Verify API tracking

## Architecture Diagram
```
[Image Generation Pipeline]
         │
         ├─ [Database Layer]
         │     ├─ SQLite Storage
         │     ├─ Connection Management
         │     └─ Error Recovery
         │
         ├─ [Component Layer]
         │     ├─ ImageEvaluator
         │     ├─ PromptHandler
         │     ├─ PromptRefiner
         │     └─ APIClient
         │
         └─ [Service Layer]
               ├─ Error Handling
               ├─ Rate Limiting
               └─ Status Tracking
```

## Contact
Original Author: Curtis Mortensen
Last Updated: [Current Date]

This should provide enough context for another AI to understand the changes, their purpose, and how to continue development or troubleshooting.