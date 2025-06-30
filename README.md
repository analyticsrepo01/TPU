# TPU Model Deployment Collection

This repository contains a collection of Jupyter notebooks demonstrating how to deploy and serve large language models (LLMs) on Google Cloud TPUs using Vertex AI and vLLM. The main focus is on the latest **tpu_v6e_with_report.ipynb** notebook, which demonstrates deployment and performance testing of Llama 3.1 8B on TPU v6e (Trillium) with comprehensive benchmarking results.

## 📚 Notebooks Overview

### 🚀 Main Notebook: tpu_v6e_with_report.ipynb
**Vertex AI Model Garden - Llama 3.1 8B Deployment with Performance Testing on TPU v6e**

- **Purpose**: Deploy and comprehensively test Llama 3.1 8B model on TPU v6e (Trillium) with detailed performance analysis
- **Key Features**:
  - Complete deployment workflow for Llama 3.1 8B on single TPU v6e
  - Comprehensive performance testing with multiple concurrency levels
  - Detailed latency and throughput benchmarking
  - Professional performance reports with CSV and Markdown outputs
  - Error handling and retry logic for production readiness
- **TPU Configuration**: 1 TPU v6e (ct6e-standard-1t)
- **Region**: europe-west4
- **Performance Results**:
  - **TTFT (Time to First Token)**: 0.276s (p95)
  - **Inter-token Latency**: 0.050s (p95)
  - **Token Output Throughput**: 1,112 tokens/sec
  - **Success Rate**: 100% (50 requests, 250 concurrent users)
  - **End-to-End Latency**: 13.8s (p95)
- **Test Configuration**:
  - Input tokens: ~82 tokens average
  - Output tokens: ~309 tokens average
  - Temperature: 0.7
  - Max tokens: 350

### 1. v3-qwen.ipynb
**Vertex AI Model Garden - Llama 3.1 and Qwen2.5 Models Deployment**

- **Purpose**: Deploy Llama 3.1 8B and Qwen2.5 1.5B models with vLLM on TPU v5e
- **Key Features**:
  - Supports both Llama 3.1 8B (requires 4 TPU v5e cores) and Qwen2.5 1.5B (requires 1 TPU v5e core)
  - Uses vLLM serving framework for improved throughput
  - Includes model access via Hugging Face with token authentication
  - Demonstrates model deployment and inference on TPU v5e
- **TPU Configuration**: 
  - Llama 3.1 8B: 4 TPU v5e (ct5lp-hightpu-4t)
  - Qwen2.5 1.5B: 1 TPU v5e (ct5lp-hightpu-1t)
- **Region**: us-central1
- **Prerequisites**: Hugging Face token for model access

### 2. v3.ipynb
**Vertex AI Model Garden - Llama 3.1 and Qwen2.5 Models Deployment (v3)**

- **Purpose**: Enhanced version of Llama 3.1 and Qwen2.5 deployment with additional model support
- **Key Features**:
  - Supports Llama 3.3 70B Instruct model configuration
  - Same TPU v5e deployment architecture as v3-qwen.ipynb
  - Enhanced environment variable handling with dotenv support
  - Improved model deployment workflow
- **TPU Configuration**: Same as v3-qwen.ipynb
- **Notable Differences**: 
  - Updated model selection to include Llama 3.3 70B Instruct
  - Enhanced configuration management

### 3. v1.ipynb
**Vertex AI Model Garden - Llama 3.1 and Qwen2.5 Models Deployment (v1)**

- **Purpose**: Original implementation of Llama 3.1 and Qwen2.5 deployment
- **Key Features**:
  - Basic deployment setup for TPU v5e
  - Qwen2.5 1.5B model focus
  - Standard vLLM configuration
  - Simple inference examples
- **TPU Configuration**: 1 TPU v5e for Qwen2.5 1.5B model
- **Status**: Baseline implementation

### 4. TPUv6_v1.ipynb
**Vertex AI Model Garden - Llama 3.1 and Qwen3 Models Deployment on TPU v6e**

- **Purpose**: Deploy Llama 3.1 8B and Qwen3 32B models on next-generation TPU v6e (Trillium)
- **Key Features**:
  - Utilizes TPU v6e (Trillium) for enhanced performance
  - Supports Llama 3.1 8B (1 TPU v6e) and Qwen3 32B (4 TPU v6e)
  - Enhanced vLLM configuration with chunked prefill and prefix caching
  - Updated serving framework with improved performance optimizations
- **TPU Configuration**:
  - Llama 3.1 8B: 1 TPU v6e (ct6e-standard-1t)
  - Qwen3 32B: 4 TPU v6e (ct6e-standard-4t)
- **Advanced Features**:
  - Chunked prefill enabled
  - Prefix caching enabled
  - VLLM_USE_V1 environment variable
  - Enhanced port configuration (7080 vs 8080)

## 🚀 Quick Start

### Prerequisites
- Google Cloud Project with billing enabled
- Vertex AI API and Compute Engine API enabled
- Appropriate TPU quotas for your region
- Hugging Face account with read access token (for Llama models)
- `.env` file with `HF_TOKEN` for Hugging Face authentication

### Common Setup Steps
1. Set up your Google Cloud environment:
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   gcloud services enable aiplatform.googleapis.com compute.googleapis.com
   ```

2. Create a `.env` file with your Hugging Face token:
   ```
   HF_TOKEN=your_hugging_face_token_here
   ```

3. Choose the appropriate notebook based on your TPU requirements:
   - **TPU v5e**: Use `v1.ipynb`, `v3.ipynb`, or `v3-qwen.ipynb`
   - **TPU v6e**: Use `TPUv6_v1.ipynb` for latest Trillium architecture

### Model Requirements

| Model | TPU Type | TPU Count | Machine Type | Notebook | Performance |
|-------|----------|-----------|--------------|----------|-------------|
| **Llama 3.1 8B** | **TPU v6e** | **1** | **ct6e-standard-1t** | **tpu_v6e_with_report.ipynb** | **1,112 tok/sec, 0.276s TTFT** |
| Qwen2.5 1.5B | TPU v5e | 1 | ct5lp-hightpu-1t | v1.ipynb, v3.ipynb, v3-qwen.ipynb | Not benchmarked |
| Llama 3.1 8B | TPU v5e | 4 | ct5lp-hightpu-4t | v3.ipynb, v3-qwen.ipynb | Not benchmarked |
| Llama 3.1 8B | TPU v6e | 1 | ct6e-standard-1t | TPUv6_v1.ipynb | Not benchmarked |
| Qwen3 32B | TPU v6e | 4 | ct6e-standard-4t | TPUv6_v1.ipynb | Not benchmarked |

## 📊 Performance Testing Results

The **tpu_v6e_with_report.ipynb** notebook includes comprehensive performance testing capabilities with detailed benchmarking results for Llama 3.1 8B on TPU v6e.

### Test Results Summary
Based on load testing with 250 concurrent users and 50 requests:

| Metric | Value | Target/Benchmark |
|--------|-------|------------------|
| **TTFT (p50)** | 0.276s | < 0.9s ✅ |
| **TTFT (p95)** | 0.277s | < 0.9s ✅ |
| **Inter-token Latency (p95)** | 0.050s | < 0.17s ✅ |
| **End-to-End Latency (p95)** | 13.8s | Variable |
| **Token Output Throughput** | 1,112 tok/sec | > 10 tok/sec ✅ |
| **Overall Token Throughput** | 1,407 tok/sec | > 1,500 tok/sec ⚠️ |
| **Requests per Second** | 3.61 req/sec | Variable |
| **Success Rate** | 100% | > 95% ✅ |

### Performance Testing Features

The notebook includes advanced performance testing capabilities:

- **Load Testing**: Configurable concurrent users and request volumes
- **Metrics Collection**: TTFT, inter-token latency, throughput measurements
- **Error Handling**: Retry logic for 502 errors and timeout handling
- **Report Generation**: Automated CSV and Markdown report generation
- **Multiple Test Scenarios**: From small-scale (5 concurrent) to high-load (250 concurrent) testing

### Test Configuration Options

```python
TEST_CONFIG = {
    'concurrent_users': 250,    # Configurable concurrency
    'total_requests': 50,       # Total test requests
    'input_token_length': 265,  # Target input length
    'output_tokens': 317,       # Target output length
    'temperature': 0.7,         # Sampling temperature
    'max_tokens': 350,          # Maximum output tokens
}
```

### Generated Reports

The performance testing generates multiple output formats:

1. **Summary CSV**: High-level metrics and aggregated results
2. **Detailed CSV**: Per-request detailed performance data
3. **Markdown Report**: Professional formatted performance analysis
4. **Console Output**: Real-time progress and results

## 🔧 Technical Details

### vLLM Configuration
- **Serving Framework**: vLLM (experimental TPU support)
- **Container**: `us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20250529_0917_tpu_experimental_RC00`
- **Max Model Length**: 2048-4096 tokens (configurable)
- **Max Running Sequences**: 256 (configurable)

### Key Features Across Notebooks
- **Environment Variables**: Secure token management via `.env` files
- **Cloud Storage**: GCS bucket integration for model artifacts
- **Endpoint Management**: Support for dedicated endpoints
- **Cleanup**: Resource cleanup utilities to manage costs

## 💰 Cost Considerations

These notebooks use billable Google Cloud components:
- **Vertex AI**: Model deployment and serving
- **Cloud Storage**: Artifact storage
- **TPU**: Compute resources for inference

Use the [Google Cloud Pricing Calculator](https://cloud.google.com/products/calculator/) to estimate costs based on your usage patterns.

## 🔍 Troubleshooting

### Common Issues
1. **Quota Errors**: Ensure you have sufficient TPU quota in your region
2. **Token Authentication**: Verify your Hugging Face token has proper read permissions
3. **Regional Availability**: Some TPU types may not be available in all regions
4. **Service Timeouts**: Reduce `max_tokens` if experiencing timeout errors

### Regional Considerations
- **Primary Region**: us-central1
- **TPU v6e Availability**: Limited regions (check current availability)
- **Bucket Region**: Must match deployment region

## 📖 Additional Resources

- [Vertex AI TPU Documentation](https://cloud.google.com/vertex-ai/docs/predictions/use-tpu)
- [vLLM on TPU Documentation](https://docs.vllm.ai/en/latest/getting_started/tpu-installation.html)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [Google Cloud Quotas](https://cloud.google.com/docs/quotas/view-manage)

## 🏷️ Tags
`vertex-ai` `tpu` `llm` `vllm` `llama` `qwen` `model-deployment` `machine-learning` `google-cloud`
