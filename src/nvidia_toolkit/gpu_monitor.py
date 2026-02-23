"""GPU Monitoring Module

This module provides comprehensive GPU monitoring capabilities using nvidia-ml-py.
Supports querying GPU information, monitoring utilization, temperature, and memory usage.
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    import pynvml
except ImportError:
    raise ImportError(
        "nvidia-ml-py is required. Install with: pip install nvidia-ml-py"
    )


@dataclass
class GPUInfo:
    """GPU device information"""
    index: int
    name: str
    uuid: str
    driver_version: str
    cuda_version: str
    total_memory: int  # in bytes
    compute_capability: Tuple[int, int]


@dataclass
class GPUMetrics:
    """Real-time GPU metrics"""
    timestamp: datetime
    gpu_index: int
    gpu_utilization: int  # percentage
    memory_used: int  # in bytes
    memory_total: int  # in bytes
    memory_utilization: int  # percentage
    temperature: int  # in celsius
    power_usage: int  # in watts
    power_limit: int  # in watts
    fan_speed: Optional[int]  # percentage, if available


class GPUMonitor:
    """GPU monitoring and information gathering utility"""
    
    def __init__(self):
        """Initialize the GPU monitor"""
        self._initialized = False
        self._gpu_count = 0
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize NVIDIA ML library"""
        try:
            pynvml.nvmlInit()
            self._gpu_count = pynvml.nvmlDeviceGetCount()
            self._initialized = True
        except pynvml.NVMLError as e:
            raise RuntimeError(f"Failed to initialize NVIDIA ML: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
    
    def shutdown(self) -> None:
        """Shutdown NVIDIA ML library"""
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
                self._initialized = False
            except pynvml.NVMLError:
                pass
    
    @property
    def gpu_count(self) -> int:
        """Get the number of available GPUs"""
        return self._gpu_count
    
    def get_driver_version(self) -> str:
        """Get NVIDIA driver version"""
        if not self._initialized:
            raise RuntimeError("GPU monitor not initialized")
        
        try:
            return pynvml.nvmlSystemGetDriverVersion()
        except pynvml.NVMLError as e:
            raise RuntimeError(f"Failed to get driver version: {e}")
    
    def get_cuda_version(self) -> str:
        """Get CUDA version"""
        if not self._initialized:
            raise RuntimeError("GPU monitor not initialized")
        
        try:
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
            major = cuda_version // 1000
            minor = (cuda_version % 1000) // 10
            return f"{major}.{minor}"
        except pynvml.NVMLError as e:
            raise RuntimeError(f"Failed to get CUDA version: {e}")
    
    def get_gpu_info(self, gpu_index: int) -> GPUInfo:
        """Get detailed information about a specific GPU"""
        if not self._initialized:
            raise RuntimeError("GPU monitor not initialized")
        
        if gpu_index >= self._gpu_count:
            raise ValueError(f"GPU index {gpu_index} out of range (0-{self._gpu_count-1})")
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            
            name = pynvml.nvmlDeviceGetName(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Get compute capability
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            
            return GPUInfo(
                index=gpu_index,
                name=name,
                uuid=uuid,
                driver_version=self.get_driver_version(),
                cuda_version=self.get_cuda_version(),
                total_memory=memory_info.total,
                compute_capability=(major, minor)
            )
        except pynvml.NVMLError as e:
            raise RuntimeError(f"Failed to get GPU info for index {gpu_index}: {e}")
    
    def get_all_gpu_info(self) -> List[GPUInfo]:
        """Get information about all available GPUs"""
        return [self.get_gpu_info(i) for i in range(self._gpu_count)]
    
    def get_gpu_metrics(self, gpu_index: int) -> GPUMetrics:
        """Get real-time metrics for a specific GPU"""
        if not self._initialized:
            raise RuntimeError("GPU monitor not initialized")
        
        if gpu_index >= self._gpu_count:
            raise ValueError(f"GPU index {gpu_index} out of range (0-{self._gpu_count-1})")
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            
            # Get utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Get memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_utilization = int((memory_info.used / memory_info.total) * 100)
            
            # Get temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError:
                temperature = 0
            
            # Get power usage
            try:
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # Convert mW to W
                power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) // 1000
            except pynvml.NVMLError:
                power_usage = 0
                power_limit = 0
            
            # Get fan speed (optional, may not be available on all GPUs)
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            except pynvml.NVMLError:
                fan_speed = None
            
            return GPUMetrics(
                timestamp=datetime.now(),
                gpu_index=gpu_index,
                gpu_utilization=utilization.gpu,
                memory_used=memory_info.used,
                memory_total=memory_info.total,
                memory_utilization=memory_utilization,
                temperature=temperature,
                power_usage=power_usage,
                power_limit=power_limit,
                fan_speed=fan_speed
            )
        except pynvml.NVMLError as e:
            raise RuntimeError(f"Failed to get GPU metrics for index {gpu_index}: {e}")
    
    def get_all_gpu_metrics(self) -> List[GPUMetrics]:
        """Get real-time metrics for all available GPUs"""
        return [self.get_gpu_metrics(i) for i in range(self._gpu_count)]
    
    def monitor_gpus(self, duration: int = 60, interval: float = 1.0) -> List[List[GPUMetrics]]:
        """Monitor all GPUs for a specified duration
        
        Args:
            duration: Monitoring duration in seconds
            interval: Sampling interval in seconds
            
        Returns:
            List of metrics samples, where each sample contains metrics for all GPUs
        """
        if not self._initialized:
            raise RuntimeError("GPU monitor not initialized")
        
        samples = []
        start_time = time.time()
        
        while (time.time() - start_time) < duration:
            sample_start = time.time()
            
            # Collect metrics for all GPUs
            gpu_metrics = self.get_all_gpu_metrics()
            samples.append(gpu_metrics)
            
            # Wait for the next interval
            elapsed = time.time() - sample_start
            if elapsed < interval:
                time.sleep(interval - elapsed)
        
        return samples
    
    def get_system_summary(self) -> Dict:
        """Get a comprehensive system summary"""
        if not self._initialized:
            raise RuntimeError("GPU monitor not initialized")
        
        gpu_info_list = self.get_all_gpu_info()
        gpu_metrics_list = self.get_all_gpu_metrics()
        
        total_memory = sum(gpu.total_memory for gpu in gpu_info_list)
        total_memory_used = sum(metrics.memory_used for metrics in gpu_metrics_list)
        avg_utilization = sum(metrics.gpu_utilization for metrics in gpu_metrics_list) / len(gpu_metrics_list) if gpu_metrics_list else 0
        avg_temperature = sum(metrics.temperature for metrics in gpu_metrics_list) / len(gpu_metrics_list) if gpu_metrics_list else 0
        total_power = sum(metrics.power_usage for metrics in gpu_metrics_list)
        
        return {
            "driver_version": self.get_driver_version(),
            "cuda_version": self.get_cuda_version(),
            "gpu_count": self._gpu_count,
            "total_memory_gb": round(total_memory / (1024**3), 2),
            "total_memory_used_gb": round(total_memory_used / (1024**3), 2),
            "memory_utilization_percent": round((total_memory_used / total_memory) * 100, 1) if total_memory > 0 else 0,
            "average_gpu_utilization_percent": round(avg_utilization, 1),
            "average_temperature_celsius": round(avg_temperature, 1),
            "total_power_usage_watts": total_power,
            "gpus": [
                {
                    "index": gpu.index,
                    "name": gpu.name,
                    "memory_gb": round(gpu.total_memory / (1024**3), 2),
                    "compute_capability": f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}",
                    "utilization_percent": gpu_metrics_list[gpu.index].gpu_utilization if gpu.index < len(gpu_metrics_list) else 0,
                    "memory_used_gb": round(gpu_metrics_list[gpu.index].memory_used / (1024**3), 2) if gpu.index < len(gpu_metrics_list) else 0,
                    "temperature_celsius": gpu_metrics_list[gpu.index].temperature if gpu.index < len(gpu_metrics_list) else 0,
                    "power_usage_watts": gpu_metrics_list[gpu.index].power_usage if gpu.index < len(gpu_metrics_list) else 0,
                }
                for gpu in gpu_info_list
            ]
        }


# Utility functions
def format_bytes(bytes_value: int) -> str:
    """Format bytes into human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def is_nvidia_gpu_available() -> bool:
    """Check if NVIDIA GPUs are available on the system"""
    try:
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return gpu_count > 0
    except (pynvml.NVMLError, Exception):
        return False