# NVIDIA GPU NIM Toolkit - Quick Start Guide

## Installation

### Option 1: Quick Install with Dependencies
```bash
cd AISecurity/nvidia-gpu-nim-toolkit
pip install -e . 
```

### Option 2: Use the Installation Script
```bash
cd AISecurity/nvidia-gpu-nim-toolkit
python install.py
```

### Option 3: Install Dependencies Manually
```bash
pip install typer rich
pip install nvidia-ml-py  # For GPU monitoring (optional)
# Install NVIDIA NIM SDK from NGC for NIM functionality (optional)
```

## Quick Test

Check if the tool is working:
```bash
nvidia-toolkit --help
nvidia-toolkit features
nvidia-toolkit system-info
```

## Available Commands

- `nvidia-toolkit features` - Check feature availability
- `nvidia-toolkit system-info` - System information
- `nvidia-toolkit gpu-status` - GPU status (requires nvidia-ml-py)
- `nvidia-toolkit gpu-info` - Detailed GPU info (requires nvidia-ml-py)
- `nvidia-toolkit nim-status` - NIM service status (requires NIM SDK)

## Example Usage

```python
# Python API
from nvidia_toolkit import GPUMonitor, NIMClient

# Check available features
from nvidia_toolkit import get_available_features
print(get_available_features())
```

## Security Features

- GitHub Actions CI/CD pipeline
- Trend Micro Application Security (TMAS) scanning
- Automated security vulnerability detection
- Container security scanning

## Project Structure

- `src/nvidia_toolkit/` - Main package code
- `examples/` - Usage examples
- `.github/workflows/` - CI/CD and security scanning
- `tests/` - Test suite (to be added)

## Development

```bash
# Install in development mode
pip install -e ."[dev]"

# Run security scan
# Automatically runs via GitHub Actions on push/PR
```

## Dependencies

**Core Dependencies (Always Required):**
- typer - CLI framework
- rich - Rich terminal output

**Optional Dependencies:**
- nvidia-ml-py - GPU monitoring (install: `pip install nvidia-ml-py`)
- NVIDIA NIM SDK - AI inference (install from NGC)

The toolkit gracefully handles missing optional dependencies and provides clear installation instructions.