# NVIDIA GPU NIM Toolkit

A comprehensive Python toolkit for GPU monitoring and NVIDIA NIM inference with integrated security scanning capabilities.

[![Security Scan](https://github.com/yourorg/nvidia-gpu-nim-toolkit/actions/workflows/security-scan.yml/badge.svg)](https://github.com/yourorg/nvidia-gpu-nim-toolkit/actions/workflows/security-scan.yml)
[![CI/CD](https://github.com/yourorg/nvidia-gpu-nim-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/yourorg/nvidia-gpu-nim-toolkit/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## Features

- 🚀 **GPU Monitoring**: Real-time GPU utilization, memory, temperature, and power monitoring
- 🤖 **NIM Integration**: Full NVIDIA NIM (NVIDIA Inference Microservice) client support
- 🔒 **Security Scanning**: Integrated with Trend Micro Application Security (TMAS)
- 📊 **Rich CLI**: Beautiful command-line interface with progress bars and tables
- 📈 **Monitoring**: Continuous GPU monitoring with data export capabilities
- 🧪 **Testing**: Comprehensive test suite with pytest
- 🐳 **Docker Ready**: Containerized deployment support

## Quick Start

### Prerequisites

- Python 3.12+
- NVIDIA GPU with drivers installed
- NVIDIA ML library (nvidia-ml-py)
- Optional: NVIDIA NIM service running

### Installation

```bash
# Clone the repository
git clone https://github.com/yourorg/nvidia-gpu-nim-toolkit.git
cd nvidia-gpu-nim-toolkit

# Install the package
pip install -e .

# Or install from PyPI (when published)
pip install nvidia-gpu-nim-toolkit
```

### Quick Commands

```bash
# Show system information
nvidia-toolkit system-info

# Check GPU status
nvidia-toolkit gpu-status

# Monitor GPUs for 60 seconds
nvidia-toolkit gpu-monitor --duration 60

# Check NIM service status
nvidia-toolkit nim-status

# Run inference (if NIM service is available)
nvidia-toolkit nim-infer llama2 "Hello, how are you?"
```

## Installation Guide

### Development Installation

```bash
# Clone and set up development environment
git clone https://github.com/yourorg/nvidia-gpu-nim-toolkit.git
cd nvidia-gpu-nim-toolkit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Production Installation

```bash
pip install nvidia-gpu-nim-toolkit
```

### Docker Installation

```bash
# Build the Docker image
docker build -t nvidia-gpu-nim-toolkit .

# Run with GPU support
docker run --gpus all -it nvidia-gpu-nim-toolkit nvidia-toolkit system-info
```

## Usage Examples

### GPU Monitoring

```python
from nvidia_toolkit import GPUMonitor

# Monitor GPUs in a context manager
with GPUMonitor() as monitor:
    # Get GPU information
    gpu_info = monitor.get_all_gpu_info()
    print(f"Found {len(gpu_info)} GPUs")
    
    # Get real-time metrics
    metrics = monitor.get_all_gpu_metrics()
    for metric in metrics:
        print(f"GPU {metric.gpu_index}: {metric.gpu_utilization}% utilization")
    
    # Monitor for a period
    samples = monitor.monitor_gpus(duration=30, interval=1.0)
    print(f"Collected {len(samples)} samples")
```

### NIM Client

```python
from nvidia_toolkit import NIMClient
from nvidia_toolkit.nim_client import create_text_inference_request

# Connect to NIM service
with NIMClient("http://localhost:8000") as client:
    # List available models
    models = client.list_models()
    print(f"Available models: {[m.name for m in models]}")
    
    # Run inference
    request = create_text_inference_request(
        model_id="llama2",
        text="Explain quantum computing",
        max_tokens=100
    )
    
    response = client.run_inference(request)
    print(f"Response: {response.outputs}")
    print(f"Latency: {response.latency_ms}ms")
```

### Command Line Interface

```bash
# GPU Commands
nvidia-toolkit gpu-info          # Detailed GPU information
nvidia-toolkit gpu-status        # Current GPU status
nvidia-toolkit gpu-monitor       # Monitor GPUs continuously

# NIM Commands
nvidia-toolkit nim-status        # NIM service status
nvidia-toolkit nim-models        # List available models
nvidia-toolkit nim-infer model "text"  # Run inference

# System Commands
nvidia-toolkit system-info       # Comprehensive system info
```

## Configuration

### Environment Variables

```bash
# NIM Service Configuration
export NIM_API_URL="http://localhost:8000"
export NIM_API_KEY="your-api-key"

# Monitoring Configuration
export GPU_MONITOR_INTERVAL=1.0
export GPU_MONITOR_DURATION=60
```

### Configuration File

Create a `config.yaml` file:

```yaml
nim:
  api_url: "http://localhost:8000"
  api_key: "your-api-key"
  timeout: 30

monitoring:
  interval: 1.0
  duration: 60
  output_dir: "./monitoring_data"

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## API Reference

### GPUMonitor

```python
class GPUMonitor:
    def __init__(self)
    def get_all_gpu_info(self) -> List[GPUInfo]
    def get_all_gpu_metrics(self) -> List[GPUMetrics]
    def monitor_gpus(self, duration: int, interval: float) -> List[List[GPUMetrics]]
    def get_system_summary(self) -> Dict
```

### NIMClient

```python
class NIMClient:
    def __init__(self, api_base_url: str, api_key: Optional[str], timeout: int)
    def list_models(self) -> List[ModelInfo]
    def run_inference(self, request: InferenceRequest) -> InferenceResponse
    def get_system_status(self) -> Dict[str, Any]
    def wait_for_model_ready(self, model_id: str) -> bool
```

## Security

This project integrates with Trend Micro Application Security (TMAS) for continuous security scanning using the official `trendmicro/tmas-scan-action@v2`:

### TMAS Security Features
- **Vulnerability Scanning**: ✅ Identifies security vulnerabilities in dependencies
- **Secrets Detection**: ✅ Scans for exposed API keys, passwords, and sensitive data  
- **Code Analysis**: Static analysis for security issues
- **Container Security**: Docker image vulnerability scanning

### Setup TMAS Integration

1. **Get TMAS API Key**: 
   - Sign up for Trend Vision One
   - Generate an API key in the TMAS console
   
2. **Configure GitHub Secrets**:
   ```
   TMAS_API_KEY=your_tmas_api_key_here
   ```

3. **Security Workflow Triggers**:
   - Push to main/develop branches
   - Pull requests to main branch
   - Manual workflow dispatch

### Additional Security Tools
- **Safety**: Python dependency vulnerability checking
- **Bandit**: Python code security analysis

## Development

### Setting Up Development Environment

```bash
# Clone and setup
git clone https://github.com/yourorg/nvidia-gpu-nim-toolkit.git
cd nvidia-gpu-nim-toolkit

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=nvidia_toolkit

# Run specific test file
pytest tests/test_gpu_monitor.py

# Run with verbose output
pytest -v
```

### Code Quality

```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/

# Run all quality checks
pre-commit run --all-files
```

### Building and Publishing

```bash
# Build package
python -m build

# Upload to test PyPI
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*
```

## Examples

Check the `examples/` directory for complete usage examples:

- `gpu_monitoring_example.py` - Comprehensive GPU monitoring
- `nim_inference_example.py` - NIM model inference
- `batch_processing_example.py` - Batch GPU processing
- `monitoring_dashboard.py` - Real-time monitoring dashboard

## Troubleshooting

### Common Issues

**NVIDIA drivers not found:**
```bash
# Check if NVIDIA drivers are installed
nvidia-smi

# Install NVIDIA drivers if missing
# Follow NVIDIA's installation guide for your OS
```

**NIM service not available:**
```bash
# Check if NIM service is running
curl http://localhost:8000/health

# Start NIM service (example)
docker run -d --gpus all -p 8000:8000 nvcr.io/nvidia/nim:latest
```

**Permission errors:**
```bash
# Add user to docker group (Linux)
sudo usermod -a -G docker $USER

# Restart shell session
```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export NVIDIA_TOOLKIT_LOG_LEVEL=DEBUG
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation for API changes
- Run security scans before submitting
- Use conventional commits for commit messages

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [NVIDIA](https://nvidia.com) for GPU computing libraries
- [Trend Micro](https://trendmicro.com) for security scanning integration
- The open-source community for various dependencies

## Support

- 📚 [Documentation](https://github.com/yourorg/nvidia-gpu-nim-toolkit/wiki)
- 🐛 [Issue Tracker](https://github.com/yourorg/nvidia-gpu-nim-toolkit/issues)
- 💬 [Discussions](https://github.com/yourorg/nvidia-gpu-nim-toolkit/discussions)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and version history.

---

**Note**: This toolkit requires NVIDIA GPUs and drivers. For CPU-only environments, some features will be limited.