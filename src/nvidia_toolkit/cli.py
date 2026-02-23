"""Command Line Interface for NVIDIA GPU NIM Toolkit"""

import json
import time
import typer
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

# Import modules conditionally to handle missing dependencies
try:
    from .gpu_monitor import GPUMonitor, is_nvidia_gpu_available, format_bytes
    GPU_MONITOR_AVAILABLE = True
except ImportError:
    GPU_MONITOR_AVAILABLE = False

try:
    from .nim_client import NIMClient, is_nim_service_available, create_text_inference_request
    NIM_CLIENT_AVAILABLE = True
except ImportError:
    NIM_CLIENT_AVAILABLE = False

app = typer.Typer(
    name="nvidia-toolkit",
    help="NVIDIA GPU NIM Toolkit - Monitor GPUs and run NIM inference",
    rich_markup_mode="rich"
)
console = Console()

def check_gpu_availability():
    """Check if GPU monitoring is available"""
    if not GPU_MONITOR_AVAILABLE:
        rprint("[red]❌ GPU monitoring not available. Install nvidia-ml-py: pip install nvidia-ml-py[/red]")
        raise typer.Exit(1)

def check_nim_availability():
    """Check if NIM client is available"""
    if not NIM_CLIENT_AVAILABLE:
        rprint("[red]❌ NIM client not available. Install NIM SDK for full functionality[/red]")
        raise typer.Exit(1)

# GPU monitoring commands
@app.command(name="gpu-info")
def gpu_info():
    """Show detailed GPU information"""
    check_gpu_availability()
    
    if not is_nvidia_gpu_available():
        rprint("[red]❌ No NVIDIA GPUs detected or drivers not installed[/red]")
        raise typer.Exit(1)
    
    with GPUMonitor() as monitor:
        gpu_infos = monitor.get_all_gpu_info()
        
        if not gpu_infos:
            rprint("[yellow]No GPUs found[/yellow]")
            return
        
        for gpu in gpu_infos:
            table = Table(title=f"GPU {gpu.index}: {gpu.name}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Name", gpu.name)
            table.add_row("UUID", gpu.uuid)
            table.add_row("Driver Version", gpu.driver_version)
            table.add_row("CUDA Version", gpu.cuda_version)
            table.add_row("Total Memory", format_bytes(gpu.total_memory))
            table.add_row("Compute Capability", f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
            
            console.print(table)
            console.print()


@app.command(name="gpu-status")
def gpu_status():
    """Show real-time GPU status"""
    check_gpu_availability()
    
    if not is_nvidia_gpu_available():
        rprint("[red]❌ No NVIDIA GPUs detected or drivers not installed[/red]")
        rprint("[blue]💡 Running in simulation mode - install NVIDIA drivers for real GPU monitoring[/blue]")
        
        # Show simulated data for demonstration
        table = Table(title="GPU Status (Simulation Mode)")
        table.add_column("GPU", style="cyan")
        table.add_column("GPU %", style="magenta")
        table.add_column("Memory %", style="blue")
        table.add_column("Memory Used", style="green")
        table.add_column("Temp (°C)", style="red")
        table.add_column("Power (W)", style="yellow")
        
        table.add_row("0", "45%", "65%", "3.2 GB / 8.0 GB", "72", "150")
        console.print(table)
        return
    
    with GPUMonitor() as monitor:
        metrics = monitor.get_all_gpu_metrics()
        
        if not metrics:
            rprint("[yellow]No GPU metrics available[/yellow]")
            return
        
        table = Table(title="GPU Status")
        table.add_column("GPU", style="cyan")
        table.add_column("GPU %", style="magenta")
        table.add_column("Memory %", style="blue")
        table.add_column("Memory Used", style="green")
        table.add_column("Temp (°C)", style="red")
        table.add_column("Power (W)", style="yellow")
        
        for metric in metrics:
            memory_used = format_bytes(metric.memory_used)
            memory_total = format_bytes(metric.memory_total)
            
            table.add_row(
                str(metric.gpu_index),
                f"{metric.gpu_utilization}%",
                f"{metric.memory_utilization}%",
                f"{memory_used} / {memory_total}",
                f"{metric.temperature}",
                f"{metric.power_usage}"
            )
        
        console.print(table)


@app.command(name="nim-status")
def nim_status(
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="NIM service URL")
):
    """Check NIM service status"""
    check_nim_availability()
    
    if not is_nim_service_available(url):
        rprint(f"[red]❌ NIM service not available at {url}[/red]")
        rprint("[blue]💡 Running in simulation mode - NIM service not detected[/blue]")
        
        # Show simulated status
        console.print(Panel(
            "✅ Service: Healthy (Simulated)",
            title="NIM Service Status (Simulation Mode)",
            border_style="green"
        ))
        
        table = Table(title="Models (Simulated)")
        table.add_column("Model ID", style="cyan")
        table.add_column("Name", style="blue")
        table.add_column("Status", style="green")
        table.add_column("Version", style="yellow")
        
        table.add_row("llama2-7b", "Llama 2 7B", "[green]ready[/green]", "1.0.0")
        table.add_row("gpt-3.5-turbo", "GPT-3.5 Turbo", "[green]ready[/green]", "0.1.0")
        
        console.print(table)
        return
    
    with NIMClient(url) as client:
        status = client.get_system_status()
        
        # Service health panel
        health_color = "green" if status.get('service_healthy') else "red"
        health_symbol = "✅" if status.get('service_healthy') else "❌"
        
        console.print(Panel(
            f"{health_symbol} Service: {'Healthy' if status.get('service_healthy') else 'Unhealthy'}",
            title="NIM Service Status",
            border_style=health_color
        ))


# System commands
@app.command(name="system-info")
def system_info():
    """Show comprehensive system information"""
    console.print(Panel("NVIDIA GPU NIM Toolkit - System Information", style="bold blue"))
    
    # Check availability of components
    rprint(f"🔍 GPU Monitoring: {'✅ Available' if GPU_MONITOR_AVAILABLE else '❌ Not Available (install nvidia-ml-py)'}")
    rprint(f"🤖 NIM Client: {'✅ Available' if NIM_CLIENT_AVAILABLE else '❌ Not Available (install NIM SDK)'}")
    
    if GPU_MONITOR_AVAILABLE:
        # Check GPU availability
        gpu_available = is_nvidia_gpu_available()
        gpu_symbol = "✅" if gpu_available else "❌"
        console.print(f"\n{gpu_symbol} NVIDIA GPU: {'Available' if gpu_available else 'Not Available'}")
        
        if gpu_available:
            with GPUMonitor() as monitor:
                summary = monitor.get_system_summary()
                
                console.print(f"🔧 Driver Version: {summary['driver_version']}")
                console.print(f"🎯 CUDA Version: {summary['cuda_version']}")
                console.print(f"🎮 GPU Count: {summary['gpu_count']}")
                console.print(f"💾 Total Memory: {summary['total_memory_gb']} GB")
                console.print(f"🔥 Average Temperature: {summary['average_temperature_celsius']}°C")
                console.print(f"⚡ Total Power: {summary['total_power_usage_watts']}W")
        else:
            console.print("\n💡 Running in simulation mode for GPU monitoring")
    
    if NIM_CLIENT_AVAILABLE:
        # Check NIM availability
        nim_available = is_nim_service_available()
        nim_symbol = "✅" if nim_available else "❌"
        console.print(f"\n{nim_symbol} NIM Service: {'Available' if nim_available else 'Not Available'}")
        
        if nim_available:
            with NIMClient() as client:
                status = client.get_system_status()
                if status.get('service_healthy'):
                    model_summary = status.get('model_summary', {})
                    console.print(f"🤖 Total Models: {model_summary.get('total_models', 0)}")
                    console.print(f"✅ Ready Models: {model_summary.get('ready_models', 0)}")
        else:
            console.print("\n💡 Running in simulation mode for NIM inference")


@app.command(name="features")
def features():
    """Show available features and their status"""
    from . import get_available_features
    
    features = get_available_features()
    
    table = Table(title="Feature Availability")
    table.add_column("Feature", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Description", style="white")
    
    feature_info = {
        "gpu_monitoring": "GPU performance monitoring and metrics collection",
        "nim_inference": "NVIDIA NIM model inference and management"
    }
    
    for feature, available in features.items():
        status = "✅ Available" if available else "❌ Unavailable"
        status_color = "green" if available else "red"
        table.add_row(
            feature.replace("_", " ").title(),
            f"[{status_color}]{status}[/{status_color}]",
            feature_info.get(feature, "")
        )
    
    console.print(table)
    
    # Installation instructions
    if not features["gpu_monitoring"]:
        console.print("\n[blue]💡 To enable GPU monitoring:[/blue]")
        console.print("   pip install nvidia-ml-py")
    
    if not features["nim_inference"]:
        console.print("\n[blue]💡 To enable NIM inference:[/blue]")
        console.print("   Install NVIDIA NIM SDK from NGC")


def main():
    """Main entry point for the CLI"""
    app()


# Standalone command functions (for pyproject.toml scripts)
def gpu_monitor():
    """Entry point for gpu-monitor command"""
    try:
        gpu_status()
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")


def nim_client():
    """Entry point for nim-client command"""
    try:
        nim_status()
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()