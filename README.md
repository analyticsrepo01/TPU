# TPU Model Deployment Collection

This repository contains a collection of Jupyter notebooks demonstrating how to deploy and serve large language models (LLMs) on Google Cloud TPUs using Vertex AI and vLLM.

## 📚 Notebooks Overview

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

| Model | TPU Type | TPU Count | Machine Type | Notebook |
|-------|----------|-----------|--------------|----------|
| Qwen2.5 1.5B | TPU v5e | 1 | ct5lp-hightpu-1t | v1.ipynb, v3.ipynb, v3-qwen.ipynb |
| Llama 3.1 8B | TPU v5e | 4 | ct5lp-hightpu-4t | v3.ipynb, v3-qwen.ipynb |
| Llama 3.1 8B | TPU v6e | 1 | ct6e-standard-1t | TPUv6_v1.ipynb |
| Qwen3 32B | TPU v6e | 4 | ct6e-standard-4t | TPUv6_v1.ipynb |

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
