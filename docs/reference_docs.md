# FAL.ai Client API Documentation

## Overview
The FAL.ai client API provides a simple interface for submitting image generation requests and handling their lifecycle. It supports real-time status updates and log streaming during the generation process.

## Basic Usage

```python
import fal_client

def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
           print(log["message"])

result = fal_client.subscribe(
    "fal-ai/fast-lightning-sdxl",
    arguments={
        "prompt": "photo of a girl smiling during a sunset, with lightnings in the background"
    },
    with_logs=True,
    on_queue_update=on_queue_update,
)
print(result)
```

## API Reference

### `fal_client.subscribe()`

Submits a request to the FAL.ai API and subscribes to status updates.

#### Parameters:

- `model_id` (str): The ID of the model to use (e.g., "fal-ai/fast-lightning-sdxl")
- `arguments` (dict): Model-specific parameters including the prompt
- `with_logs` (bool): Enable log streaming during generation
- `on_queue_update` (callable): Callback function for status updates

#### Status Update Types:

- `fal_client.InProgress`: Request is being processed
- `fal_client.Completed`: Request completed successfully
- `fal_client.Failed`: Request failed with an error

### Example Status Handler

```python
def handle_status(update):
    if isinstance(update, fal_client.InProgress):
        # Handle progress updates
        print("Processing...")
        for log in update.logs:
            print(f"Log: {log['message']}")
    
    elif isinstance(update, fal_client.Completed):
        # Handle completion
        print("Generation completed!")
        
    elif isinstance(update, fal_client.Failed):
        # Handle errors
        print(f"Error: {update.error}")
```

## Common Parameters

### Image Generation

```python
arguments = {
    "prompt": str,              # Main generation prompt
    "negative_prompt": str,     # Elements to avoid in generation
    "num_inference_steps": int, # Number of denoising steps (default: 30)
    "guidance_scale": float,    # How closely to follow prompt (default: 7.5)
    "width": int,              # Image width (default: 1024)
    "height": int,             # Image height (default: 1024)
}
```

## Error Handling

The API uses exceptions to handle errors:

- `fal_client.AuthenticationError`: Invalid API credentials
- `fal_client.RequestError`: Invalid request parameters
- `fal_client.APIError`: Server-side errors

```python
try:
    result = fal_client.subscribe(...)
except fal_client.AuthenticationError:
    print("Invalid API key")
except fal_client.RequestError as e:
    print(f"Invalid request: {e}")
except fal_client.APIError as e:
    print(f"API error: {e}")
```

## Best Practices

1. Always handle potential errors
2. Use status updates for better UX
3. Enable logging during development
4. Store API keys securely
5. Implement appropriate timeouts

## Rate Limiting

The API implements rate limiting:
- Requests per minute: Varies by subscription
- Concurrent requests: Based on plan
- Automatic retry with exponential backoff

## Example Implementation

```python
import fal_client
import time

def generate_image(prompt, max_retries=3):
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            return fal_client.subscribe(
                "fal-ai/fast-lightning-sdxl",
                arguments={"prompt": prompt},
                with_logs=True,
                on_queue_update=lambda u: print(f"Status: {type(u).__name__}")
            )
        except fal_client.APIError as e:
            retry_count += 1
            if retry_count == max_retries:
                raise
            time.sleep(2 ** retry_count)  # Exponential backoff

    return None
```
