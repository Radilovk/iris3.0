# AI Model Upgrade Guide

## Summary of Changes

This upgrade replaces the older GPT-4o (2024) model with support for newer, more suitable AI models available in 2026:

### Default Configuration (Recommended)
- **Provider:** Google Gemini
- **Model:** `gemini-2.0-flash-exp`
- **Advantages:**
  - Latest experimental model with superior vision capabilities
  - 1 million token context window (vs 128K)
  - Native multimodal support
  - Faster processing speed
  - More cost-effective
  - Free tier available for testing

### Alternative Options

#### Option 1: GPT-4o-mini (Cost-effective OpenAI)
```toml
AI_PROVIDER = "openai"
AI_MODEL = "gpt-4o-mini"
```
- 15-25x cheaper than GPT-4o
- Strong vision capabilities (59.4% MMMU)
- Best for high-volume, budget-conscious deployments

#### Option 2: GPT-4o (Premium OpenAI)
```toml
AI_PROVIDER = "openai"
AI_MODEL = "gpt-4o"
```
- Keep existing model
- Premium pricing
- Excellent vision and reasoning

## Migration Steps

### 1. Update Configuration

Edit `wrangler.toml`:
```toml
[vars]
AI_PROVIDER = "gemini"  # or "openai"
AI_MODEL = "gemini-2.0-flash-exp"  # or your chosen model
```

### 2. Get API Key

**For Gemini (recommended):**
1. Visit: https://aistudio.google.com/app/apikey
2. Create a new API key
3. Set it as a secret:
   ```bash
   wrangler secret put AI_API_KEY
   ```

**For OpenAI:**
1. Visit: https://platform.openai.com/api-keys
2. Create a new API key
3. Set it as a secret:
   ```bash
   wrangler secret put AI_API_KEY
   ```

### 3. Deploy

```bash
wrangler deploy
```

## Testing

### Test with Gemini

1. Set configuration:
   ```toml
   AI_PROVIDER = "gemini"
   AI_MODEL = "gemini-2.0-flash-exp"
   ```

2. Deploy and test with a sample iris image

3. Check response format matches expected JSON structure

### Test with OpenAI

1. Set configuration:
   ```toml
   AI_PROVIDER = "openai"
   AI_MODEL = "gpt-4o-mini"  # or "gpt-4o"
   ```

2. Deploy and test with the same iris image

3. Compare results and performance

## Why Upgrade?

### Technical Improvements

1. **Better Vision Analysis:**
   - Gemini 2.0 Flash can analyze up to 3,000 images per prompt
   - Native multimodal processing (not separate vision + text models)
   - Improved accuracy on medical imaging tasks

2. **Cost Efficiency:**
   - Gemini: More cost-effective than GPT-4o
   - GPT-4o-mini: 15-25x cheaper than GPT-4o ($0.15/M vs $2.50-5/M input tokens)

3. **Performance:**
   - Gemini 2.0 Flash: Optimized for speed with 1M context window
   - Better suited for the 5-step iris analysis pipeline

4. **Future-Proof:**
   - Using 2026 models instead of 2024 models
   - Access to latest AI capabilities

### Benchmark Comparison

| Metric | Gemini 2.0 Flash | GPT-4o | GPT-4o-mini |
|--------|------------------|---------|-------------|
| Vision (MMMU) | ~65% | 69.1% | 59.4% |
| Context Window | 1M tokens | 128K | 128K |
| Speed | Very Fast | Fast | Very Fast |
| Cost | Low | High | Very Low |
| Multi-image | 3000/prompt | Limited | Limited |

## Troubleshooting

### Issue: JSON Parsing Errors

Both providers return JSON in different formats:
- **OpenAI:** Returns JSON in `choices[0].message.content`
- **Gemini:** Returns JSON in `candidates[0].content.parts[0].text`

The code handles both formats automatically via `safeParseJSON()`.

### Issue: Rate Limiting

- **Gemini:** Free tier has generous limits for testing
- **OpenAI:** May require paid account for production use

### Issue: Different Response Quality

If you notice different quality between providers:
1. Try adjusting `temperature` parameter
2. Test with multiple sample images
3. Consider if the model needs different prompt engineering

## Rollback Instructions

If you need to rollback to the original GPT-4o configuration:

1. Edit `wrangler.toml`:
   ```toml
   AI_PROVIDER = "openai"
   AI_MODEL = "gpt-4o"
   AI_BASE_URL = "https://api.openai.com/v1"
   ```

2. Ensure you have OpenAI API key set:
   ```bash
   wrangler secret put AI_API_KEY
   ```

3. Deploy:
   ```bash
   wrangler deploy
   ```

## Support

For issues or questions:
1. Check Cloudflare Worker logs
2. Verify API key is set correctly
3. Test API endpoint manually with curl
4. Check provider-specific documentation:
   - Gemini: https://ai.google.dev/gemini-api/docs
   - OpenAI: https://platform.openai.com/docs

## Recommendations

### For Production Use
- **Recommended:** Gemini 2.0 Flash (`gemini-2.5-flash-latest` for stable production)
- **Alternative:** GPT-4o-mini for cost-effective OpenAI option

### For Testing/Development
- **Recommended:** Gemini 2.0 Flash Experimental (`gemini-2.0-flash-exp`)
- Free tier available for testing

### For Premium Quality
- **Option:** GPT-4o if budget allows and OpenAI ecosystem is preferred
- Consider if marginal quality improvement justifies 15-25x cost increase
