# Security Policy

## Automated Security Scanning

This repository uses **Trend Micro Application Security (TMAS)** for comprehensive security scanning.

### What TMAS Scans

#### 🐳 **Container Image Vulnerability Scanning**
TMAS automatically scans the Docker container image for security vulnerabilities:

**Container Components:**
- `nvidia/cuda:12.2.0-runtime-ubuntu22.04` - NVIDIA CUDA base image
- Ubuntu 22.04 system packages and libraries
- Python 3.11 runtime and installed packages
- NVIDIA GPU libraries and drivers
- Application dependencies and configurations

**Security Checks:**
- 🔍 **Base Image CVEs** - Known vulnerabilities in Ubuntu and CUDA layers
- 📦 **Package Vulnerabilities** - Security issues in installed system packages
- 🛡️ **Malware Detection** - Scanning for malicious code in container layers
- 🕵️ **Secrets Scanning** - Detection of exposed credentials or keys
- 🔒 **Configuration Analysis** - Container security best practices

#### 🔍 **Dependency Vulnerability Scanning**
TMAS automatically scans all Python packages in `requirements.txt` for known vulnerabilities:

**Core Dependencies:**
- `nvidia-ml-py>=12.0.0` - NVIDIA GPU monitoring library
- `click>=8.0.0` - Command line interface creation toolkit
- `rich>=13.0.0` - Rich text and beautiful formatting
- `typer>=0.9.0` - Modern CLI framework
- `requests>=2.28.0` - HTTP library
- `pydantic>=2.0.0` - Data validation and settings management
- `psutil>=5.9.0` - System and process utilities

**Development Dependencies:**
- `pytest>=7.0.0` - Testing framework
- `pytest-cov>=4.0.0` - Coverage testing
- `black>=23.0.0` - Code formatter
- `flake8>=6.0.0` - Code linting
- `mypy>=1.0.0` - Static type checking
- `pre-commit>=3.0.0` - Git hooks framework

#### 🕵️ **Secrets Scanning**
- API keys, tokens, and credentials in source code
- Configuration files and scripts
- Hardcoded passwords or sensitive information

#### 🐛 **Code Vulnerability Scanning**
- Source code analysis for security vulnerabilities
- Python-specific security issues
- Common web vulnerabilities and security anti-patterns

### What TMAS Excludes

To optimize scanning performance and avoid false positives, TMAS excludes:
- `.git/` directory and Git metadata files
- Compiled Python files (`*.pyc`, `*.pyo`)
- Binary files (`*.so`, `*.dll`, `*.dylib`)
- Cache directories (`__pycache__/`, `.pytest_cache/`)
- Build artifacts and temporary files
- IDE configuration files

### Scan Triggers

TMAS scans are triggered on:
- Every push to the repository
- Pull requests to main branch
- Manual workflow dispatch

### Viewing Results

Scan results are available in:
- GitHub Actions workflow logs
- Security tab in the GitHub repository
- SARIF reports (if enabled)

### Reporting Security Issues

If you discover a security vulnerability, please report it by:
1. Creating a private security advisory in GitHub
2. Emailing the maintainers directly
3. Using responsible disclosure practices

### Dependencies Security Updates

We regularly update dependencies to ensure:
- Latest security patches are applied
- Known vulnerabilities are addressed
- Compatibility with security tools

### Security Best Practices

This project follows security best practices:
- ✅ Automated dependency vulnerability scanning
- ✅ Secrets detection and prevention
- ✅ Static code analysis
- ✅ Regular security updates
- ✅ Minimal privilege principles
- ✅ Secure coding guidelines

---

*Security scanning powered by Trend Micro Application Security (TMAS)*