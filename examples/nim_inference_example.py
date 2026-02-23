#!/usr/bin/env python3
"""
NIM Inference Example

This example demonstrates NVIDIA NIM inference capabilities
using the nvidia-gpu-nim-toolkit.
"""

import time
import asyncio
from datetime import datetime
from nvidia_toolkit import NIMClient, is_nim_service_available
from nvidia_toolkit.nim_client import create_text_inference_request, ModelStatus

def main():
    """Main function demonstrating NIM inference features"""
    
    print("🤖 NVIDIA NIM Inference Example")
    print("=" * 40)
    
    # Configuration
    nim_url = "http://localhost:8000"  # Change this to your NIM service URL
    
    # Check if NIM service is available
    if not is_nim_service_available(nim_url):
        print(f"❌ NIM service not available at {nim_url}")
        print("   Please ensure NVIDIA NIM service is running")
        print("   Example: docker run -d --gpus all -p 8000:8000 nvcr.io/nvidia/nim:latest")
        return
    
    # Initialize NIM client
    with NIMClient(nim_url) as client:
        # Check service health
        print("\n🔍 Service Health Check:")
        print("-" * 30)
        health = client.health_check()
        print(f"   Status: {'✅ Healthy' if health.get('status') == 'healthy' else '❌ Unhealthy'}")
        
        # Get service information
        try:
            service_info = client.get_service_info()
            print(f"   Service Version: {service_info.get('version', 'Unknown')}")
        except Exception as e:
            print(f"   Service Info: Could not retrieve ({e})")
        
        # List available models
        print("\n📋 Available Models:")
        print("-" * 25)
        models = client.list_models()
        
        if not models:
            print("   No models available")
            print("   Please deploy a model first")
            return
        
        ready_models = [m for m in models if m.status == ModelStatus.READY]
        
        for model in models:
            status_emoji = {
                ModelStatus.READY: "✅",
                ModelStatus.LOADING: "⏳", 
                ModelStatus.ERROR: "❌",
                ModelStatus.STOPPED: "⏹️"
            }.get(model.status, "❓")
            
            print(f"   {status_emoji} {model.name} ({model.model_id})")
            print(f"     Version: {model.version}")
            print(f"     Status: {model.status.value}")
            if model.description:
                print(f"     Description: {model.description}")
            print()
        
        if not ready_models:
            print("❌ No models are ready for inference")
            print("   Please wait for models to load or check model deployment")
            return
        
        # Select the first ready model for demonstration
        selected_model = ready_models[0]
        print(f"🎯 Selected Model: {selected_model.name} ({selected_model.model_id})")
        
        # Single inference example
        print("\n💬 Single Inference Example:")
        print("-" * 35)
        
        # Create test prompts
        test_prompts = [
            "What is artificial intelligence?",
            "Explain machine learning in simple terms.",
            "What are the benefits of GPU acceleration?"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n   Test {i}: {prompt}")
            
            try:
                # Create inference request
                request = create_text_inference_request(
                    model_id=selected_model.model_id,
                    text=prompt,
                    max_tokens=100,
                    temperature=0.7
                )
                
                # Run inference
                start_time = time.time()
                response = client.run_inference(request)
                end_time = time.time()
                
                # Display results
                print(f"   ✅ Response ({response.latency_ms:.1f}ms):")
                response_text = response.outputs.get('text', str(response.outputs))
                # Truncate long responses for display
                if len(response_text) > 200:
                    response_text = response_text[:200] + "..."
                print(f"   📝 {response_text}")
                
            except Exception as e:
                print(f"   ❌ Inference failed: {e}")
        
        # Batch inference example
        print("\n🔄 Batch Inference Example:")
        print("-" * 30)
        
        batch_prompts = [
            "Define neural networks.",
            "What is deep learning?",
            "Explain computer vision.",
            "What is natural language processing?"
        ]
        
        batch_requests = [
            create_text_inference_request(
                model_id=selected_model.model_id,
                text=prompt,
                max_tokens=50,
                temperature=0.5
            )
            for prompt in batch_prompts
        ]
        
        print(f"   Running {len(batch_requests)} inference requests...")
        
        start_time = time.time()
        batch_responses = client.run_batch_inference(batch_requests)
        total_time = time.time() - start_time
        
        print(f"   ✅ Batch completed in {total_time:.2f}s")
        print(f"   📊 Average latency: {sum(r.latency_ms for r in batch_responses)/len(batch_responses):.1f}ms")
        
        # Display batch results
        for i, (prompt, response) in enumerate(zip(batch_prompts, batch_responses), 1):
            if 'error' not in response.outputs:
                response_text = response.outputs.get('text', str(response.outputs))
                if len(response_text) > 100:
                    response_text = response_text[:100] + "..."
                print(f"   {i}. {prompt[:30]}... → {response_text}")
            else:
                print(f"   {i}. {prompt[:30]}... → ❌ Error: {response.outputs['error']}")
        
        # Performance monitoring
        print("\n⚡ Performance Monitoring:")
        print("-" * 30)
        
        try:
            metrics = client.get_model_metrics(selected_model.model_id)
            if 'error' not in metrics:
                print("   📈 Model Metrics:")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"     {key}: {value}")
            else:
                print(f"   ❌ Metrics unavailable: {metrics.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"   ❌ Could not retrieve metrics: {e}")
        
        # System status
        print("\n🖥️ System Status:")
        print("-" * 20)
        system_status = client.get_system_status()
        
        if system_status.get('service_healthy'):
            model_summary = system_status.get('model_summary', {})
            print(f"   Total Models: {model_summary.get('total_models', 0)}")
            print(f"   Ready: {model_summary.get('ready_models', 0)}")
            print(f"   Loading: {model_summary.get('loading_models', 0)}")
            print(f"   Errors: {model_summary.get('error_models', 0)}")
        else:
            print("   ❌ System unhealthy")
        
        # Model deployment example (if needed)
        if len([m for m in models if m.status == ModelStatus.STOPPED]) > 0:
            print("\n🚀 Model Deployment Example:")
            print("-" * 35)
            
            stopped_models = [m for m in models if m.status == ModelStatus.STOPPED]
            example_model = stopped_models[0]
            
            print(f"   Example: Deploy {example_model.name}")
            print("   (This is just an example - not actually deploying)")
            
            # Uncomment to actually deploy:
            # try:
            #     deploy_result = client.deploy_model(example_model.model_id)
            #     print(f"   ✅ Deployment initiated: {deploy_result}")
            #     
            #     # Wait for model to be ready
            #     if client.wait_for_model_ready(example_model.model_id, max_wait_time=300):
            #         print(f"   ✅ Model {example_model.model_id} is ready!")
            #     else:
            #         print(f"   ⏱️ Model deployment timed out")
            # except Exception as e:
            #     print(f"   ❌ Deployment failed: {e}")
    
    print("\n✅ NIM inference examples completed!")
    print("\n💡 Tips:")
    print("   - Adjust temperature for creativity vs consistency")
    print("   - Use max_tokens to control response length")
    print("   - Monitor latency for performance optimization")
    print("   - Use batch inference for better throughput")


if __name__ == "__main__":
    main()