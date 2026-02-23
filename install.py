#!/usr/bin/env python3
"""
NVIDIA GPU NIM Toolkit Installation Script

This script helps install the toolkit with proper dependency handling.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, check=True, capture_output=False):
    """Run a shell command with error handling"""
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result
        else:
            subprocess.run(cmd, shell=True, check=check)
            return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"   Error: {e}")
        return False

def check_nvidia_drivers():
    """Check if NVIDIA drivers are available"""
    result = run_command("nvidia-smi", check=False, capture_output=True)
    return result.returncode == 0

def install_base_requirements():
    """Install base requirements (excluding NVIDIA-specific packages)"""
    print("📦 Installing base requirements...")
    
    base_packages = [
        "click>=8.0.0",
        "rich>=13.0.0", 
        "typer>=0.9.0",
        "requests>=2.28.0",
        "pydantic>=2.0.0",
        "psutil>=5.9.0"
    ]
    
    for package in base_packages:
        print(f"   Installing {package}...")
        if not run_command(f"pip install '{package}'", check=False):
            print(f"   ⚠️  Warning: Failed to install {package}")
        else:
            print(f"   ✅ Installed {package}")

def install_nvidia_packages():
    """Install NVIDIA-specific packages if drivers are available"""
    if not check_nvidia_drivers():
        print("⚠️  NVIDIA drivers not detected. Skipping nvidia-ml-py installation.")
        print("   The toolkit will run in simulation mode.")
        return False
    
    print("🎯 NVIDIA drivers detected. Installing nvidia-ml-py...")
    if run_command("pip install 'nvidia-ml-py>=12.0.0'", check=False):
        print("   ✅ nvidia-ml-py installed successfully")
        return True
    else:
        print("   ❌ Failed to install nvidia-ml-py")
        return False

def install_development_packages():
    """Install development packages"""
    print("🛠️  Installing development packages...")
    
    dev_packages = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0", 
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0"
    ]
    
    for package in dev_packages:
        run_command(f"pip install '{package}'", check=False)

def install_package():
    """Install the nvidia-toolkit package in development mode"""
    print("📦 Installing nvidia-toolkit package...")
    if run_command("pip install -e .", check=False):
        print("   ✅ nvidia-toolkit installed in development mode")
        return True
    else:
        print("   ❌ Failed to install nvidia-toolkit package")
        return False

def main():
    """Main installation process"""
    print("🚀 NVIDIA GPU NIM Toolkit Installation")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required. Current version:", sys.version)
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Change to project directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Install base requirements
    install_base_requirements()
    
    # Try to install NVIDIA packages
    nvidia_available = install_nvidia_packages()
    
    # Install the package
    package_installed = install_package()
    
    if not package_installed:
        print("❌ Installation failed!")
        return False
    
    print("\n🎉 Installation Summary:")
    print("=" * 30)
    print(f"   ✅ Base packages: Installed")
    print(f"   {'✅' if nvidia_available else '⚠️ '} NVIDIA support: {'Available' if nvidia_available else 'Simulation mode'}")
    print(f"   ✅ nvidia-toolkit: Installed")
    
    print("\n📚 Next Steps:")
    print("   1. Try the examples:")
    print("      python examples/gpu_monitoring_example.py")
    print("      python examples/nim_inference_example.py")
    print("   2. Use the CLI:")
    print("      nvidia-toolkit --help")
    print("   3. Read the README.md for more information")
    
    if not nvidia_available:
        print("\n💡 Note: Running in simulation mode since NVIDIA drivers not detected.")
        print("   Install NVIDIA drivers and run this script again for full functionality.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)