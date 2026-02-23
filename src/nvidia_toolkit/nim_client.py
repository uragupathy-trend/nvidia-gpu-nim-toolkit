"""NVIDIA NIM Client Module

This module provides a client for interacting with NVIDIA NIM (NVIDIA Inference Microservice).
Supports model deployment, inference requests, and status monitoring.
"""

import json
import time
import requests
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

try:
    import nimlib
    import nim_sdk
except ImportError:
    raise ImportError(
        "nim-sdk and nimlib are required. Install with: pip install nim-sdk nimlib"
    )


class ModelStatus(Enum):
    """Model deployment status"""
    UNKNOWN = "unknown"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class ModelInfo:
    """Model information"""
    model_id: str
    name: str
    version: str
    status: ModelStatus
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class InferenceRequest:
    """Inference request data"""
    model_id: str
    inputs: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = 30


@dataclass
class InferenceResponse:
    """Inference response data"""
    request_id: str
    model_id: str
    outputs: Dict[str, Any]
    latency_ms: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class NIMClient:
    """NVIDIA NIM client for inference operations"""
    
    def __init__(self, 
                 api_base_url: str = "http://localhost:8000",
                 api_key: Optional[str] = None,
                 timeout: int = 30):
        """Initialize NIM client
        
        Args:
            api_base_url: Base URL for NIM API
            api_key: Optional API key for authentication
            timeout: Default request timeout in seconds
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set up authentication if API key is provided
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}'
            })
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'nvidia-gpu-nim-toolkit/0.1.0'
        })
    
    def _make_request(self, 
                     method: str, 
                     endpoint: str, 
                     data: Optional[Dict] = None,
                     params: Optional[Dict] = None,
                     timeout: Optional[int] = None) -> requests.Response:
        """Make HTTP request to NIM API"""
        url = f"{self.api_base_url}{endpoint}"
        request_timeout = timeout or self.timeout
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=request_timeout
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"NIM API request failed: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check NIM service health"""
        try:
            response = self._make_request('GET', '/health')
            return response.json()
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get NIM service information"""
        response = self._make_request('GET', '/info')
        return response.json()
    
    def list_models(self) -> List[ModelInfo]:
        """List available models"""
        response = self._make_request('GET', '/models')
        models_data = response.json()
        
        models = []
        for model_data in models_data.get('models', []):
            status_str = model_data.get('status', 'unknown').lower()
            try:
                status = ModelStatus(status_str)
            except ValueError:
                status = ModelStatus.UNKNOWN
            
            model = ModelInfo(
                model_id=model_data.get('id', ''),
                name=model_data.get('name', ''),
                version=model_data.get('version', ''),
                status=status,
                description=model_data.get('description'),
                created_at=self._parse_datetime(model_data.get('created_at')),
                updated_at=self._parse_datetime(model_data.get('updated_at'))
            )
            models.append(model)
        
        return models
    
    def get_model_info(self, model_id: str) -> ModelInfo:
        """Get information about a specific model"""
        response = self._make_request('GET', f'/models/{model_id}')
        model_data = response.json()
        
        status_str = model_data.get('status', 'unknown').lower()
        try:
            status = ModelStatus(status_str)
        except ValueError:
            status = ModelStatus.UNKNOWN
        
        return ModelInfo(
            model_id=model_data.get('id', ''),
            name=model_data.get('name', ''),
            version=model_data.get('version', ''),
            status=status,
            description=model_data.get('description'),
            created_at=self._parse_datetime(model_data.get('created_at')),
            updated_at=self._parse_datetime(model_data.get('updated_at'))
        )
    
    def deploy_model(self, model_id: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        """Deploy a model
        
        Args:
            model_id: ID of the model to deploy
            config: Optional deployment configuration
            
        Returns:
            Deployment response data
        """
        data = {'model_id': model_id}
        if config:
            data['config'] = config
        
        response = self._make_request('POST', '/models/deploy', data=data)
        return response.json()
    
    def stop_model(self, model_id: str) -> Dict[str, Any]:
        """Stop a deployed model"""
        response = self._make_request('POST', f'/models/{model_id}/stop')
        return response.json()
    
    def run_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference on a model
        
        Args:
            request: Inference request data
            
        Returns:
            Inference response
        """
        start_time = time.time()
        
        # Prepare request data
        data = {
            'inputs': request.inputs
        }
        if request.parameters:
            data['parameters'] = request.parameters
        
        # Make inference request
        response = self._make_request(
            'POST', 
            f'/models/{request.model_id}/infer',
            data=data,
            timeout=request.timeout
        )
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        response_data = response.json()
        
        return InferenceResponse(
            request_id=response_data.get('request_id', ''),
            model_id=request.model_id,
            outputs=response_data.get('outputs', {}),
            latency_ms=latency_ms,
            timestamp=datetime.now(),
            metadata=response_data.get('metadata')
        )
    
    def run_batch_inference(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Run batch inference
        
        Args:
            requests: List of inference requests
            
        Returns:
            List of inference responses
        """
        responses = []
        for req in requests:
            try:
                response = self.run_inference(req)
                responses.append(response)
            except Exception as e:
                # Create error response
                error_response = InferenceResponse(
                    request_id='',
                    model_id=req.model_id,
                    outputs={'error': str(e)},
                    latency_ms=0.0,
                    timestamp=datetime.now(),
                    metadata={'error': True}
                )
                responses.append(error_response)
        
        return responses
    
    def get_model_metrics(self, model_id: str) -> Dict[str, Any]:
        """Get metrics for a specific model"""
        try:
            response = self._make_request('GET', f'/models/{model_id}/metrics')
            return response.json()
        except Exception as e:
            return {
                'error': str(e),
                'model_id': model_id,
                'timestamp': datetime.now().isoformat()
            }
    
    def wait_for_model_ready(self, 
                           model_id: str, 
                           max_wait_time: int = 300,
                           check_interval: int = 5) -> bool:
        """Wait for a model to become ready
        
        Args:
            model_id: Model ID to wait for
            max_wait_time: Maximum wait time in seconds
            check_interval: Check interval in seconds
            
        Returns:
            True if model becomes ready, False if timeout
        """
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait_time:
            try:
                model_info = self.get_model_info(model_id)
                if model_info.status == ModelStatus.READY:
                    return True
                elif model_info.status == ModelStatus.ERROR:
                    raise RuntimeError(f"Model {model_id} failed to deploy")
                
                time.sleep(check_interval)
            except Exception as e:
                raise RuntimeError(f"Failed to check model status: {e}")
        
        return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            health = self.health_check()
            service_info = self.get_service_info()
            models = self.list_models()
            
            model_summary = {
                'total_models': len(models),
                'ready_models': len([m for m in models if m.status == ModelStatus.READY]),
                'loading_models': len([m for m in models if m.status == ModelStatus.LOADING]),
                'error_models': len([m for m in models if m.status == ModelStatus.ERROR]),
            }
            
            return {
                'service_healthy': health.get('status') == 'healthy',
                'service_info': service_info,
                'model_summary': model_summary,
                'models': [
                    {
                        'id': m.model_id,
                        'name': m.name,
                        'status': m.status.value,
                        'version': m.version
                    }
                    for m in models
                ],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'service_healthy': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _parse_datetime(self, datetime_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string"""
        if not datetime_str:
            return None
        
        try:
            # Try ISO format first
            return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None
    
    def close(self):
        """Close the client session"""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Utility functions
def create_text_inference_request(model_id: str, 
                                text: str, 
                                max_tokens: int = 100,
                                temperature: float = 0.7) -> InferenceRequest:
    """Create a text generation inference request"""
    return InferenceRequest(
        model_id=model_id,
        inputs={'text': text},
        parameters={
            'max_tokens': max_tokens,
            'temperature': temperature
        }
    )


def create_image_inference_request(model_id: str,
                                 image_data: Union[str, bytes],
                                 format: str = 'base64') -> InferenceRequest:
    """Create an image inference request"""
    if isinstance(image_data, bytes):
        import base64
        image_data = base64.b64encode(image_data).decode('utf-8')
    
    return InferenceRequest(
        model_id=model_id,
        inputs={
            'image': image_data,
            'format': format
        }
    )


def is_nim_service_available(api_base_url: str = "http://localhost:8000") -> bool:
    """Check if NIM service is available"""
    try:
        client = NIMClient(api_base_url)
        health = client.health_check()
        return health.get('status') == 'healthy'
    except Exception:
        return False