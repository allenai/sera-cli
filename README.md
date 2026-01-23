# Ai2 Soft-Verified Efficient Repository Agents (SERA) Claude Code Proxy

This repo allows Claude Code to be used with the Ai2 Open Coding Agents SERA model.

You will need [Claude Code](https://code.claude.com/docs/en/overview) and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed to set up the SERA CLI.

We support either using SERA via third party endpoints or by self-hosting on Modal. After the model is running, the `sera` executable lets you use Claude Code with the model.

## Self-Hosted Modal Deployment

The model can also be deployed Modal. This takes ~10m for the first run as ~65GB of model weights are downloaded. Subsequent runs will cache the model and will start up faster.
When you exit Claude Code, the modal app will automatically get cleaned up.

```bash
# First, install modal and sera globally
uv tool install modal
uv tool install ai2-sera-cli

# Setup modal. This will prompt you to set up an account
modal setup

# Deploy SERA to Modal and launch Claude Code
sera --modal
```

## Using Shared Endpoints

If you have endpoints available for the SERA model, you can use them through the following steps:

```bash
# Install sera globally
uv tool install ai2-sera-cli

# Export the API key your endpoints if applicable
export SERA_API_KEY=<your API key>

# Run Sera
sera --endpoint <endpoint URL>
```

## Configuration

**CLI Options:**
- `--endpoint URL` - vLLM endpoint URL (required unless `--modal` is used)
- `--modal` - Deploy vLLM to Modal instead of using local/remote endpoint
- `--port PORT` - Proxy server port (default: 8080)
- `--model MODEL` - Model name/path
- `--hf-secret NAME` - Modal secret name containing `HF_TOKEN` for private/gated models
- `--proxy-only` - Start proxy only, don't launch Claude Code

### API Key Configuration

The proxy supports API key authentication for vLLM endpoints:

- **Local vLLM endpoints**: Set the `SERA_API_KEY` environment variable to provide an API key for authentication.
  ```bash
  export SERA_API_KEY=your_api_key_here
  sera --endpoint http://localhost:6767/v1/chat/completions
  ```

- **Modal deployments**: API keys are automatically generated as UUIDs when deploying to Modal.

- **vLLM API key**: If your vLLM server requires an API key, you can set it via the `--api-key` flag when starting vLLM, or set the `VLLM_API_KEY` environment variable before starting the proxy.

The proxy automatically includes the API key in the `Authorization: Bearer <api_key>` header when making requests to the vLLM endpoint.

### Using Private HuggingFace Models

To use private or gated models from HuggingFace with Modal deployment:

```bash
# 1. Create a Modal secret with your HuggingFace token
modal secret create huggingface HF_TOKEN=hf_your_token_here

# 2. Deploy with the secret and your private model
sera --modal --hf-secret huggingface --model your-org/private-model
```

