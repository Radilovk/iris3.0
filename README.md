# iris3.0

Iris iridology analysis system using advanced AI vision models for medical image analysis.

## Overview

This project uses cutting-edge AI models to analyze unwrapped iris images for iridology diagnosis. The system supports multiple AI providers with modern, high-performance vision models optimized for medical image analysis.

## AI Model Configuration (2026)

### Supported AI Providers

The system now supports multiple AI providers with state-of-the-art vision models:

#### 1. **Google Gemini** (Recommended - Default)
- **Model:** `gemini-2.0-flash-exp` (Latest experimental)
- **Advantages:**
  - Native multimodal capabilities with superior image analysis
  - 1 million token context window
  - Can analyze up to 3,000 images per prompt
  - Faster processing speed
  - Built-in native image generation
  - More cost-effective than GPT-4o
  - Free tier available for testing
- **Alternative Models:**
  - `gemini-2.5-flash-latest` - Production-ready, stable
  - `gemini-1.5-pro-latest` - Fallback option

#### 2. **OpenAI GPT-4o**
- **Model:** `gpt-4o` (Premium flagship)
- **Advantages:**
  - Excellent vision and reasoning capabilities
  - Full multimodal support (text, images, audio, video)
  - High benchmark scores (MMMU 69.1)
  - 128K context window
- **Cost:** Premium pricing ($2.50-5/M input tokens)

#### 3. **OpenAI GPT-4o-mini** (Cost-effective)
- **Model:** `gpt-4o-mini`
- **Advantages:**
  - 15-25x cheaper than GPT-4o ($0.15/M input tokens)
  - Strong vision capabilities (MMMU 59.4%)
  - 128K context window
  - Excellent for high-volume processing
- **Best for:** Budget-conscious deployments, batch processing

### Configuration

Edit `wrangler.toml` to configure your AI provider:

```toml
[vars]
# Choose provider: "openai", "gemini", or "openai-compatible"
AI_PROVIDER = "gemini"

# Set your model
AI_MODEL = "gemini-2.0-flash-exp"  # or "gpt-4o", "gpt-4o-mini"

# API URLs (configured automatically based on provider)
AI_BASE_URL = "https://api.openai.com/v1"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta"
```

### Setting API Keys

For **Gemini** (recommended):
```bash
# Get your API key from: https://aistudio.google.com/app/apikey
wrangler secret put AI_API_KEY
```

For **OpenAI**:
```bash
# Get your API key from: https://platform.openai.com/api-keys
wrangler secret put AI_API_KEY
```

## Deployment

1. Install Wrangler:
   ```bash
   npm install -g wrangler
   ```

2. Log in:
   ```bash
   wrangler login
   ```

3. Create KV namespace:
   ```bash
   wrangler kv namespace create IRIS_KV
   ```
   Copy the `id` and `preview_id` from output into `wrangler.toml`

4. Set your AI API key:
   ```bash
   wrangler secret put AI_API_KEY
   ```

5. Deploy:
   ```bash
   wrangler deploy
   ```

## Usage

### Analyze Iris Image
```bash
POST https://<worker-subdomain>.workers.dev/analyze
```

Form fields:
- `strip_image` — base64 JPEG from app.py
- `side` — "R" or "L"
- `image_hash` — optional unique ID
- `questionnaire` — optional JSON with patient data

### Retrieve Results
```bash
GET https://<worker-subdomain>.workers.dev/result/<side>:<hash>
```

## Model Comparison

| Feature | Gemini 2.0 Flash | GPT-4o | GPT-4o-mini |
|---------|------------------|---------|-------------|
| Vision Quality | Excellent | Excellent | Very Good |
| Speed | Very Fast | Fast | Very Fast |
| Context Window | 1M tokens | 128K | 128K |
| Cost | Low | High | Very Low |
| Multi-image | 3000/prompt | Limited | Limited |
| Native JSON | Yes | Yes | Yes |
| **Recommended** | ✅ Yes | For premium needs | For budget |

## Why Upgrade from GPT-4o?

The original configuration used `gpt-4o` from 2024. In 2026, better options are available:

1. **Gemini 2.0 Flash** offers:
   - Comparable or better vision analysis
   - Much faster processing
   - Lower costs
   - Larger context window (1M vs 128K)
   - Better suited for medical image analysis

2. **GPT-4o-mini** offers:
   - Significantly lower costs (15-25x cheaper)
   - Strong vision capabilities
   - Better for high-volume deployments

## Architecture

The system uses a 5-step AI pipeline:
- **STEP1:** Geo calibration
- **STEP2A/2B:** Structural + pigment detection (parallel)
- **STEP2C:** Consistency validator
- **STEP3:** Zone mapper
- **STEP4:** Profile builder
- **STEP5:** Bulgarian report generation

All steps leverage the configured AI model's vision capabilities for precise iris analysis.