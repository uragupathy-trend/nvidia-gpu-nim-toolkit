"""NVIDIA GPU NIM Toolkit

A comprehensive toolkit for GPU monitoring and NVIDIA NIM inference 
with integrated security scanning capabilities.
"""

__version__ = "0.1.0"
__author__ = "AI Security Team"

# Import modules conditionally to handle missing dependencies gracefully
try:
    from .gpu_monitor import GPUMonitor
    _GPU_MONITOR_AVAILABLE = True
except ImportError:
    GPUMonitor = None
    _GPU_MONITOR_AVAILABLE = False

try:
    from .nim_client import NIMClient
    _NIM_CLIENT_AVAILABLE = True
except ImportError:
    NIMClient = None
    _NIM_CLIENT_AVAILABLE = False

# Build __all__ dynamically based on what's available
__all__ = []
if _GPU_MONITOR_AVAILABLE:
    __all__.append("GPUMonitor")
if _NIM_CLIENT_AVAILABLE:
    __all__.append("NIMClient")

def get_available_features():
    """Get a list of available features based on installed dependencies."""
    features = {
        "gpu_monitoring": _GPU_MONITOR_AVAILABLE,
        "nim_inference": _NIM_CLIENT_AVAILABLE,
    }
    return features