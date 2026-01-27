"""
Deploy a persistent vLLM model to Modal for multi-user setups.

This script deploys a model to Modal and keeps it running so multiple users
can call it via `sera --endpoint <URL>`. Unlike `sera --modal` which creates
ephemeral deployments that stop when you exit, `deploy-sera` creates persistent
deployments that stay up until explicitly stopped.

Usage:
    deploy-sera --model allenai/SERA-32B                    # Deploy with 1 GPU
    deploy-sera --model allenai/SERA-32B --num-gpus 2       # Deploy with 2 GPUs (tensor parallel)
    deploy-sera --model org/private-model --hf-secret huggingface  # Private model
    deploy-sera --stop                                    # Stop the running deployment

Examples:
    # Deploy a model with API key authentication
    deploy-sera --model allenai/SERA-32B --api-key mykey123

    # Use from another machine
    SERA_API_KEY=mykey123 sera --endpoint https://xxx.modal.run/v1/chat/completions

"""

from __future__ import annotations

import argparse
import os
import secrets
import subprocess
import sys
import time
from dataclasses import dataclass

import modal

# ============ Modal Configuration ============

MODAL_APP_NAME = "sera-deploy-vllm"
MODAL_VLLM_PORT = 8000
MODAL_MAX_MODEL_LEN = 32768
MODAL_GPU = "H100"
MODAL_VOLUME_NAME = "sera-demo-models"
MODAL_MODELS_DIR = "/models"
DEFAULT_MODEL = "allenai/SERA-32B"
DEFAULT_NUM_GPUS = 1


@dataclass
class Config:
    """Configuration for the deployment."""

    model: str = DEFAULT_MODEL
    num_gpus: int = DEFAULT_NUM_GPUS
    api_key: str | None = None
    hf_secret: str | None = None


# Global config populated from CLI args
CONFIG = Config()

modal_app = modal.App(MODAL_APP_NAME)
modal_volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)
modal_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "vllm==0.13.0",
    "setuptools",
    "huggingface_hub[hf_xet]",
)


def _get_modal_model() -> str:
    """Get modal model name from env (set by Modal secret) or CONFIG fallback."""
    return os.environ.get("MODAL_MODEL", CONFIG.model)


def _get_num_gpus() -> int:
    """Get number of GPUs from env (set by Modal secret) or CONFIG fallback."""
    return int(os.environ.get("MODAL_NUM_GPUS", CONFIG.num_gpus))


def _get_model_local_path() -> str:
    """Get the local path where the model should be stored in the volume."""
    # Convert "allenai/SERA-32B" -> "/models/SERA-32B"
    model_name = _get_modal_model().split("/")[-1]
    return f"{MODAL_MODELS_DIR}/{model_name}"


def _ensure_model_downloaded() -> str:
    """Download model to volume if not already present. Returns local path."""
    from huggingface_hub import snapshot_download

    local_path = _get_model_local_path()

    if os.path.exists(local_path) and os.listdir(local_path):
        print(f"Model already cached at {local_path}")
        return local_path

    modal_model = _get_modal_model()
    print(f"Downloading {modal_model} to {local_path}...")
    print("(This only happens once - subsequent runs will use the cached model)")

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Using HuggingFace token from Modal secret")

    snapshot_download(
        repo_id=modal_model,
        local_dir=local_path,
        ignore_patterns=["*.md", "*.txt"],
        token=hf_token,
    )

    modal_volume.commit()
    print(f"Model downloaded and cached at {local_path}")
    return local_path


def modal_vllm_server():
    """Start vLLM server - Modal routes traffic to it."""
    # Ensure model is downloaded (uses cached version if available)
    model_path = _ensure_model_downloaded()
    num_gpus = _get_num_gpus()

    # Point vLLM cache to the volume for persistent compilation artifacts
    # This caches torch.compile outputs, CUDA graphs, etc. across cold starts
    cache_dir = f"{MODAL_MODELS_DIR}/.vllm_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = cache_dir

    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(MODAL_VLLM_PORT),
        "--max-model-len",
        str(MODAL_MAX_MODEL_LEN),
        "--compilation-config",
        '{"cudagraph_capture_sizes": [1, 2, 4, 8]}',
        "--max-num-seqs",
        "4",
        "--trust-remote-code",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]

    # Add tensor parallelism if using multiple GPUs
    if num_gpus > 1:
        cmd.extend(["--tensor-parallel-size", str(num_gpus)])

    # Use the HuggingFace model ID as the served name so users don't need to know about /models/ path
    cmd.extend(["--served-model-name", _get_modal_model()])

    # Add API key authentication if configured
    vllm_api_key = os.environ.get("VLLM_API_KEY")
    if vllm_api_key:
        cmd.extend(["--api-key", vllm_api_key])

    print(f"Starting vLLM with model: {model_path}")
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs with tensor parallelism")
    subprocess.Popen(cmd)


def register_modal_function() -> None:
    """Register the Modal vLLM server function with configuration from CONFIG."""
    # Pass configuration to Modal via environment variables
    env_dict = {
        "MODAL_MODEL": CONFIG.model,
        "MODAL_NUM_GPUS": str(CONFIG.num_gpus),
    }
    if CONFIG.api_key:
        env_dict["VLLM_API_KEY"] = CONFIG.api_key

    secrets = [modal.Secret.from_dict(env_dict)]
    if CONFIG.hf_secret:
        secrets.append(modal.Secret.from_name(CONFIG.hf_secret))

    # Configure GPU - for multi-GPU, use a string like "H100:2"
    gpu_config = MODAL_GPU if CONFIG.num_gpus == 1 else f"{MODAL_GPU}:{CONFIG.num_gpus}"

    # Apply decorators to register the Modal function
    modal_app.function(
        image=modal_image,
        gpu=gpu_config,
        timeout=3600,
        scaledown_window=300,
        volumes={MODAL_MODELS_DIR: modal_volume},
        secrets=secrets,
    )(
        modal.concurrent(max_inputs=100)(
            modal.web_server(port=MODAL_VLLM_PORT, startup_timeout=600)(
                modal_vllm_server
            )
        )
    )


def wait_for_vllm_ready(
    base_url: str, api_key: str | None = None, timeout: int = 1200
) -> bool:
    """Wait for vLLM server to be ready by polling /v1/models endpoint.

    Args:
        base_url: The base URL of the vLLM server (e.g., https://xxx.modal.run)
        api_key: Optional API key for authentication
        timeout: Maximum time to wait in seconds (default 20 minutes)

    Returns:
        True if server is ready, False if timeout exceeded
    """
    import httpx

    models_url = f"{base_url}/v1/models"
    start_time = time.time()

    print("Waiting for vLLM to be ready", end="", flush=True)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    while time.time() - start_time < timeout:
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(models_url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    # Check that at least one model is loaded
                    if data.get("data") and len(data["data"]) > 0:
                        elapsed = int(time.time() - start_time)
                        print(f" ready! (took {elapsed}s)")
                        return True
        except Exception:
            pass  # Server not ready yet

        print(".", end="", flush=True)
        time.sleep(3)

    print()
    print(f"Timeout: vLLM server not ready after {timeout}s")
    return False


def stop_modal_app() -> None:
    """Stop the Modal app."""
    print(f"Stopping Modal app '{MODAL_APP_NAME}'...")
    result = subprocess.run(
        ["modal", "app", "stop", MODAL_APP_NAME],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("Modal server stopped.")
    else:
        print(f"Failed to stop: {result.stderr}")
        sys.exit(1)


def deploy() -> None:
    """Deploy the vLLM model to Modal."""
    # Register the Modal function with current configuration
    register_modal_function()

    print(f"Deploying model '{CONFIG.model}' to Modal...")
    print(f"  GPUs: {CONFIG.num_gpus} x {MODAL_GPU}")
    if CONFIG.num_gpus > 1:
        print(f"  Tensor parallelism: {CONFIG.num_gpus}")
    if CONFIG.hf_secret:
        print(f"  HuggingFace secret: {CONFIG.hf_secret}")
    if CONFIG.api_key:
        print("  API key: (set)")
    print()
    print("(This may take a few minutes on first run while the container builds)")
    print()

    try:
        with modal.enable_output():
            modal_app.deploy()
    except Exception as e:
        print(f"Failed to deploy Modal app: {e}")
        sys.exit(1)

    # Get the endpoint URL
    vllm_fn = modal.Function.from_name(MODAL_APP_NAME, "modal_vllm_server")
    vllm_url = vllm_fn.get_web_url()
    if vllm_url is None:
        print("Error: Could not get vLLM endpoint URL from Modal.")
        sys.exit(1)
    assert vllm_url is not None  # for type checker

    # Wait for vLLM to be fully ready (model loaded)
    print()
    if not wait_for_vllm_ready(vllm_url, api_key=CONFIG.api_key):
        print("Error: vLLM server failed to become ready")
        sys.exit(1)

    # Print the endpoint information
    endpoint_url = f"{vllm_url}/v1/chat/completions"

    print()
    print("=" * 60)
    print("Deployment successful!")
    print("=" * 60)
    print()
    print(f"Model: {CONFIG.model}")
    print(f"Endpoint: {vllm_url}")
    print()
    print("To use with sera proxy:")
    if CONFIG.api_key:
        print(
            f"  SERA_API_KEY={CONFIG.api_key} sera --endpoint {endpoint_url} --model {CONFIG.model}"
        )
    else:
        print(f"  sera --endpoint {endpoint_url} --model {CONFIG.model}")
    print()
    print("To stop the deployment:")
    print("  deploy-sera --stop")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Deploy a vLLM model to Modal for use with the sera proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Deploy with default model (1 GPU)
    deploy-sera --model allenai/SERA-32B

    # Deploy with 2 GPUs for larger models
    deploy-sera --model allenai/SERA-32B --num-gpus 2

    # Deploy with API key authentication
    deploy-sera --model allenai/SERA-32B --api-key mysecretkey

    # Deploy a private HuggingFace model
    deploy-sera --model your-org/private-model --hf-secret huggingface

    # Stop the running deployment
    deploy-sera --stop

To create a HuggingFace secret for private models:
    modal secret create huggingface HF_TOKEN=hf_xxxxx
        """,
    )
    parser.add_argument(
        "--model",
        default=CONFIG.model,
        help=f"HuggingFace model ID to deploy (default: {CONFIG.model})",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=CONFIG.num_gpus,
        help=f"Number of GPUs to use (also sets tensor parallelism, default: {CONFIG.num_gpus})",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for vLLM authentication (users will need SERA_API_KEY env var)",
    )
    parser.add_argument(
        "--hf-secret",
        default=None,
        help=(
            "Modal secret name containing HF_TOKEN for private/gated models. "
            "Create with: modal secret create <name> HF_TOKEN=<your-token>"
        ),
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the running Modal deployment",
    )

    args = parser.parse_args()

    if args.stop:
        stop_modal_app()
        return

    CONFIG.model = args.model
    CONFIG.num_gpus = args.num_gpus
    CONFIG.hf_secret = args.hf_secret

    if args.api_key:
        CONFIG.api_key = args.api_key
    else:
        CONFIG.api_key = secrets.token_urlsafe(32)
        print(f"Generated API key: {CONFIG.api_key}")
        print()

    deploy()


if __name__ == "__main__":
    main()
