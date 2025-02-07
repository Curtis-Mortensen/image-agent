import os
import json
import requests
from typing import Dict, Any, Optional
from pathlib import Path

class FalClient:
    """Client for interacting with FAL.AI's ComfyUI API."""
    
    BASE_URL = "https://110602490-comfyui-sd-xl.fal.run"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the FAL.AI client.
        
        Args:
            api_key (str, optional): FAL.AI API key. If not provided, will look for FAL_API_KEY env variable.
        """
        self.api_key = api_key or os.getenv("FAL_API_KEY")
        if not self.api_key:
            raise ValueError("FAL.AI API key must be provided or set as FAL_API_KEY environment variable")
        
        self.headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def load_workflow(self, workflow_path: str) -> Dict[str, Any]:
        """
        Load a ComfyUI workflow from a JSON file.
        
        Args:
            workflow_path (str): Path to the workflow JSON file
            
        Returns:
            dict: Loaded workflow configuration
        """
        path = Path(workflow_path)
        if not path.exists():
            raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
            
        with open(path, 'r') as f:
            return json.load(f)
    
    def generate_image(self, workflow: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an image using the FAL.AI ComfyUI API.
        
        Args:
            workflow (dict): ComfyUI workflow configuration
            inputs (dict): Input parameters for the workflow
            
        Returns:
            dict: API response containing generated image data
        """
        payload = {
            "workflow": workflow,
            "inputs": inputs
        }
        
        try:
            response = requests.post(
                f"{self.BASE_URL}/v1/realtime_inference",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise FalAPIError(f"Failed to generate image: {str(e)}")
    
    def check_status(self, request_id: str) -> Dict[str, Any]:
        """
        Check the status of an ongoing generation request.
        
        Args:
            request_id (str): ID of the generation request
            
        Returns:
            dict: Status response from the API
        """
        try:
            response = requests.get(
                f"{self.BASE_URL}/v1/realtime_inference/{request_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise FalAPIError(f"Failed to check status: {str(e)}")

class FalAPIError(Exception):
    """Custom exception for FAL.AI API errors."""
    pass