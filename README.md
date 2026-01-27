# Ai2 Soft-Verified Efficient Repository Agents (SERA) Claude Code Proxy

This repo allows Claude Code to be used with the Ai2 Open Coding Agents SERA model.

You will need [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed to set up the SERA CLI.

For more information, see the [Open Coding Agents Blog Post](https://allenai.org/blog/open-coding-agents), [SERA Training Code](https://github.com/allenai/SERA), [SERA Paper](https://allenai.org/papers/opencodingagents), and [Ai2 Open Coding Agents Hugging Face Collection](https://huggingface.co/collections/allenai/open-coding-agents).

## Quick Start with Modal

The fastest way to try SERA is with Modal, which handles GPU provisioning, vLLM deployment, and downloading the model automatically. This takes ~10m for the first run as ~65GB of model weights are downloaded. Subsequent runs will cache the model and start up faster.

When you exit Claude Code, the Modal app will automatically get cleaned up.

```bash
# Install modal and sera globally
uv tool install modal
uv tool install ai2-sera-cli

# Setup modal (this will prompt you to set up an account)
modal setup

# Deploy SERA to Modal and launch Claude Code
sera --modal
```

## Using Existing Endpoints

If you have an existing vLLM endpoint for the SERA model (e.g., from a shared deployment or your own infrastructure):

```bash
# Install sera globally
uv tool install ai2-sera-cli

# Set the API key if your endpoint requires authentication
export SERA_API_KEY=<your API key>

# Run sera with your endpoint
sera --endpoint <endpoint URL>
```

## Shared Deployments with `deploy-sera`

For teams or multi-user setups, you can create a persistent vLLM deployment on Modal using `deploy-sera`. Unlike `sera --modal` which creates ephemeral deployments that stop when you exit, `deploy-sera` creates persistent deployments that stay up until explicitly stopped.

```bash
# Deploy a persistent vLLM instance
deploy-sera --model allenai/SERA-32B

# The command outputs an endpoint URL and API key
# Share these with your team members

# Team members can then connect with:
SERA_API_KEY=<api-key> sera --endpoint <endpoint-url>

# Stop the deployment when done
deploy-sera --stop
```

### `deploy-sera` Options

| Option             | Description                                                      |
|--------------------|------------------------------------------------------------------|
| `--model MODEL`    | HuggingFace model ID to deploy (default: `allenai/SERA-32B`)       |
| `--num-gpus N`     | Number of GPUs to use; also sets tensor parallelism (default: 1) |
| `--api-key KEY`    | API key for authentication (auto-generated if not specified)     |
| `--hf-secret NAME` | Modal secret containing `HF_TOKEN` for private/gated models      |
| `--stop`           | Stop the running deployment                                      |

### Deploying Private Models

For private models (e.g., fine-tuned on a proprietary codebase), use `--hf-secret` to authenticate with HuggingFace:

```bash
# 1. Create a Modal secret with your HuggingFace token
modal secret create huggingface HF_TOKEN=hf_your_token_here

# 2. Deploy your private model
deploy-sera --model your-org/private-sera-model --hf-secret huggingface

# 3. Users connect with the provided endpoint and API key
SERA_API_KEY=<api-key> sera --endpoint <endpoint-url>
```

For ephemeral single-user deployments, the same `--hf-secret` flag works with `sera --modal`.

## Self-Hosted vLLM

You can run SERA with vLLM on any cloud GPU provider or your own hardware directly with vLLM.

**On the server:**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model allenai/SERA-32B \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 32768 \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

**On your dev machine:**

```bash
uv tool install ai2-sera-cli
sera --endpoint http://your-server:8000/v1/chat/completions
```

## Configuration

### `sera` CLI Options

| Option             | Description                                                      |
|--------------------|------------------------------------------------------------------|
| `--endpoint URL`   | vLLM endpoint URL (required unless `--modal` is used)            |
| `--modal`          | Deploy vLLM to Modal (ephemeral, auto-cleanup on exit)           |
| `--port PORT`      | Proxy server port (default: 8080)                                |
| `--model MODEL`    | Model name/path                                                  |
| `--hf-secret NAME` | Modal secret name containing `HF_TOKEN` for private/gated models |
| `--proxy-only`     | Start proxy only, don't launch Claude Code                       |

### Environment Variables

| Variable         | Description                                            |
|------------------|--------------------------------------------------------|
| `SERA_API_KEY`   | API key for vLLM endpoint authentication               |
| `SERA_MODEL`     | Default model name (fallback for `--model`)            |
| `SERA_HF_SECRET` | Default Modal secret name (fallback for `--hf-secret`) |

### API Key Authentication

The proxy supports API key authentication for vLLM endpoints:

- **`sera --modal`**: API key is auto-generated and managed in the background
- **`deploy-sera`**: API key is auto-generated and printed so it can be shared with team members
- **Existing endpoints**: Set `SERA_API_KEY` environment variable before running `sera`
- **Self-hosted vLLM**: Start vLLM with `--api-key YOUR_KEY`, then set `SERA_API_KEY=YOUR_KEY`

The proxy includes the API key in the `Authorization: Bearer <api_key>` header when making requests.

## Citation
```
@article{sera2026,
  title={SERA: Soft-Verified Efficient Repository Agents},
  author={Shen, Ethan and Tormoen, Daniel and Shah, Saurabh and Farhadi, Ali and Dettmers, Tim},
  year={2026},
  institution={Allen Institute for AI},
  url={https://allenai.org/papers/opencodingagents}
}
```
