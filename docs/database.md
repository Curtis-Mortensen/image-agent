Here's a revised database design optimized for independent microservices and simplified tracking:

# Revised Database System Documentation

## Overview
This optimized database design focuses on prompt evolution tracking, image quality assessment, and process transparency while maintaining service isolation capabilities. Key improvements include versioned prompt tracking, unified status management, and direct quality metrics.

For each program:
    PromptImporter imports prompts into the 'prompts' table, setting status to 'imported'.
    PromptCompleter checks for prompts with 'imported' or 'incomplete' status:
        - Successfully completed prompts get status 'completed'
        - Failed completion attempts get status 'incomplete'
    BatchGenerator looks for prompts marked 'completed' or 'needs_refinement', generates images, updates status to 'generated'.
    ImageEvaluater processes images with 'generated' status, updates adherence scores and flags, sets status to 'evaluated'.
    PromptRefiner checks evaluations, refines prompts, creates new prompt versions if needed, and updates statuses.


## Database Structure (v2.0.0)

### Core Tables

1. prompts 
Tracks all prompt versions and their metadata:
- `scene_hash`: VARCHAR(64) PRIMARY KEY (SHA-256 hash of normalized scene description)
- `scene_text`: TEXT (Original scene text)
- `id`: UUID primary key
- `first_prompt_id`: UUID (First prompt that used this scene)
- `model`: VARCHAR(32) (e.g., "flux", "flux-1", "flux-2")
- `prompt_text`: Current prompt text (required)
- `base_parameters`: JSON field for model/config settings
- `generation_iteration`: Current iteration number (starts at 0)
- `status`: ENUM('draft', 'imported', 'incomplete', 'completed', 'generating', 'evaluating', 'needs_refinement', 'archived')
- `best_image_id`: UUID reference to top-rated image
- `current_score`: Numeric adherence score (0-1)
- `created_at`: Timestamp
- `last_updated`: Timestamp

2. image_evaluations
- `id`: UUID PRIMARY KEY
- `image_id`: UUID REFERENCES generated_images(id)
- `prompt_id`: UUID REFERENCES prompts(id)
- `scene_hash`: VARCHAR(64) REFERENCES scenes(scene_hash)
- `vision_model`: VARCHAR(32) (e.g., "CLIP-L14", "Fuyu-8B")
- `model_output`: JSON:
{
  "description": "A yellow cat sitting on a blue rug",
  "confidence_scores": {
    "objects": 0.92,
    "colors": 0.88,
    "spatial": 0.75
  },
  "detected_artifacts": ["extra limbs", "floating objects"]
}
- `adherence_analysis`: JSON:
{
  "text_match_score": 0.68,
  "missing_elements": ["vintage camera", "sunlight beams"],
  "unwanted_elements": ["modern furniture"],
  "style_deviation": 0.45
}  
- `created_at`: TIMESTAMP

#### 3. prompt_learnings
Tracks evolutionary knowledge between prompt versions:
- `id`: UUID primary key
- `prompt_id`: UUID reference to originating prompt version
- `parent_learning_id`: UUID reference to previous learning (chainable)
- `adherence_analysis`: JSON: (from image_evaluations)
{
  "text_match_score": 0.68,
  "missing_elements": ["vintage camera", "sunlight beams"],
  "unwanted_elements": ["modern furniture"],
  "style_deviation": 0.45
}  
- `error_patterns`: JSON object storing:
{
  "error_patterns": {
    "failed_terms": ["spherical", "metallic"],
    "common_artifacts": ["extra limbs", "texture bleeding"],
    "style_violations": ["watercolor", "impressionism"],
    "score_distribution": {"min": 0.32, "max": 0.61, "avg": 0.48}
  }
}
- `status`: ENUM('active', 'superseded', 'archived')
- `created_at`: Timestamp

### Support Tables

#### 4. generated_images
Records all generation attempts:
- `id`: UUID primary key
- `prompt_id`: UUID reference to prompt version
- `image_path`: Unique storage path (hashed naming recommended)
- `generation_metadata`: JSON with model/timestamp/parameters
- `is_adhering`: Boolean flag from evaluator
- `adherence_score`: Numeric rating (0-1) from ML evaluation
- `evaluation_details`: JSON with scoring breakdown
- `created_at`: Timestamp

#### 5. api_logs
Now includes learning references:
- `id`: UUID primary key
- `learning_id`: UUID reference to associated learning
- `error_context`: JSON with error details
- (retains previous fields with improved metadata)

```