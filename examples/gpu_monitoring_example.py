#!/usr/bin/env python3
"""
GPU Monitoring Example

This example demonstrates comprehensive GPU monitoring capabilities
using the nvidia-gpu-nim-toolkit.
"""

import time
import json
from datetime import datetime
from nvidia_toolkit import GPUMonitor, is_nvidia_gpu_available

def main():
    """Main function demonstrating GPU monitoring features"""
    
    print("🚀 NVIDIA GPU Monitoring Example")
    print("=" * 50)
    
    # Check if NVIDIA GPUs are available
    if not is_nvidia_gpu_available():
        print("❌ No NVIDIA GPUs detected or drivers not installed")
        print("   Please ensure NVIDIA GPU drivers are properly installed")
        return
    
    # Initialize GPU monitor
    with GPUMonitor() as monitor:
        # Get system overview
        print("\n📊 System Overview:")
        print("-" * 30)
        summary = monitor.get_system_summary()
        print(f"   Driver Version: {summary['driver_version']}")
        print(f"   CUDA Version: {summary['cuda_version']}")
        print(f"   GPU Count: {summary['gpu_count']}")
        print(f"   Total Memory: {summary['total_memory_gb']} GB")
        print(f"   Average Temperature: {summary['average_temperature_celsius']}°C")
        
        # Get detailed GPU information
        print("\n🎮 GPU Information:")
        print("-" * 30)
        gpu_infos = monitor.get_all_gpu_info()
        for gpu in gpu_infos:
            print(f"   GPU {gpu.index}: {gpu.name}")
            print(f"     Memory: {gpu.total_memory // (1024**3)} GB")
            print(f"     Compute: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
            print(f"     UUID: {gpu.uuid[:8]}...")
            print()
        
        # Real-time monitoring
        print("📈 Real-time Monitoring (10 samples):")
        print("-" * 40)
        print("Time      GPU  Util%  Mem%  Temp°C  Power(W)")
        
        for i in range(10):
            metrics = monitor.get_all_gpu_metrics()
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            for metric in metrics:
                print(f"{timestamp}  {metric.gpu_index:3d}  {metric.gpu_utilization:5d}  "
                      f"{metric.memory_utilization:4d}  {metric.temperature:6d}  "
                      f"{metric.power_usage:8d}")
            
            if i < 9:  # Don't sleep on last iteration
                time.sleep(2)
        
        # Extended monitoring period
        print("\n🔍 Extended Monitoring (30 seconds):")
        print("-" * 40)
        print("Starting extended monitoring...")
        
        samples = monitor.monitor_gpus(duration=30, interval=2.0)
        
        print(f"✅ Collected {len(samples)} samples")
        
        # Analyze the collected data
        if samples:
            print("\n📊 Analysis:")
            print("-" * 20)
            
            # Calculate averages for first GPU
            first_gpu_data = [sample[0] for sample in samples if sample]
            avg_gpu_util = sum(s.gpu_utilization for s in first_gpu_data) / len(first_gpu_data)
            avg_memory_util = sum(s.memory_utilization for s in first_gpu_data) / len(first_gpu_data)
            avg_temp = sum(s.temperature for s in first_gpu_data) / len(first_gpu_data)
            avg_power = sum(s.power_usage for s in first_gpu_data) / len(first_gpu_data)
            
            print(f"   Average GPU Utilization: {avg_gpu_util:.1f}%")
            print(f"   Average Memory Utilization: {avg_memory_util:.1f}%")
            print(f"   Average Temperature: {avg_temp:.1f}°C")
            print(f"   Average Power Usage: {avg_power:.1f}W")
            
            # Find peak values
            peak_gpu = max(s.gpu_utilization for s in first_gpu_data)
            peak_memory = max(s.memory_utilization for s in first_gpu_data)
            peak_temp = max(s.temperature for s in first_gpu_data)
            peak_power = max(s.power_usage for s in first_gpu_data)
            
            print(f"   Peak GPU Utilization: {peak_gpu}%")
            print(f"   Peak Memory Utilization: {peak_memory}%")
            print(f"   Peak Temperature: {peak_temp}°C")
            print(f"   Peak Power Usage: {peak_power}W")
        
        # Save monitoring data
        output_file = f"gpu_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        monitoring_data = {
            'system_summary': summary,
            'gpu_info': [
                {
                    'index': gpu.index,
                    'name': gpu.name,
                    'total_memory_gb': gpu.total_memory // (1024**3),
                    'compute_capability': f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}",
                    'uuid': gpu.uuid
                }
                for gpu in gpu_infos
            ],
            'monitoring_samples': [
                [
                    {
                        'timestamp': metric.timestamp.isoformat(),
                        'gpu_index': metric.gpu_index,
                        'gpu_utilization': metric.gpu_utilization,
                        'memory_utilization': metric.memory_utilization,
                        'temperature': metric.temperature,
                        'power_usage': metric.power_usage
                    }
                    for metric in sample
                ]
                for sample in samples
            ],
            'generated_at': datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(monitoring_data, f, indent=2)
        
        print(f"\n💾 Monitoring data saved to: {output_file}")
    
    print("\n✅ Monitoring completed successfully!")


if __name__ == "__main__":
    main()