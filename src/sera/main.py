"""
SWE-agent <-> Claude Code translation proxy.

Routes requests to a vLLM endpoint, translates between Anthropic and OpenAI API formats,
and maps SWE-agent tools (str_replace_editor, bash) to Claude Code tools (Read, Edit, Write, Bash).

Usage:
    sera --endpoint URL              # Start proxy and launch Claude Code
    sera --endpoint URL --proxy-only # Start proxy server only (prints command to run Claude Code)
    sera --modal                     # Deploy vLLM to Modal and launch Claude Code
    sera --modal --proxy-only        # Deploy vLLM to Modal (proxy only)

Modal mode:
    When --modal is used, the script will:
    1. Deploy vLLM to Modal (requires 'modal' package and authentication)
    2. Download model weights to a Modal Volume (only on first run - cached thereafter)
    3. Start the local proxy pointing to the Modal endpoint
    4. Launch Claude Code automatically (unless --proxy-only is used)
    5. Stop the Modal deployment on exit (Ctrl+C or when Claude Code exits)

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx
import modal
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

LOG_FILE = "/tmp/cc-proxy.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_ENDPOINT = "http://localhost:6767/v1/chat/completions"
DEFAULT_PORT = 8080
DEFAULT_MAX_CONTEXT_LENGTH = 32768
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that can interact with a computer to solve tasks."
)


@dataclass
class Config:
    """Configuration for the proxy."""

    endpoint: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    port: int = DEFAULT_PORT
    max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    api_key: str | None = None


# Global config populated from CLI args
CONFIG = Config()

# Stores {tool_call_id: {"swe_name": ..., "swe_input": ...}}
pending_tool_calls: dict[str, dict[str, Any]] = {}

# Cache for formatted tool results: {tool_call_id: formatted_content}
# This allows us to reuse formatting for historical messages without re-reading files
tool_result_cache: dict[str, str] = {}


def parse_context_overflow_error(error_text: str) -> tuple[int, int] | None:
    """Parse vLLM context overflow error to extract actual token counts.

    Returns (input_tokens, max_context_length) if this is a context overflow error,
    or None if it's a different type of error.

    Example errors:
    1. 'max_tokens' is too large: 8192. This model's maximum context length is 32768
       tokens and your request has 24828 input tokens
    2. This model's maximum context length is 32768 tokens. However, your request
       has 33110 input tokens. Please reduce the length of the input messages.
    """
    # Pattern 1: "and your request has X input tokens"
    match = re.search(
        r"maximum context length is (\d+) tokens and your request has (\d+) input tokens",
        error_text,
    )
    if match:
        max_context = int(match.group(1))
        input_tokens = int(match.group(2))
        return (input_tokens, max_context)

    # Pattern 2: "However, your request has X input tokens"
    match = re.search(
        r"maximum context length is (\d+) tokens\. However, your request has (\d+) input tokens",
        error_text,
    )
    if match:
        max_context = int(match.group(1))
        input_tokens = int(match.group(2))
        return (input_tokens, max_context)

    return None


def generate_error_sse_events(
    error_message: str, model: str, input_tokens: int = 0
) -> list[str]:
    """Generate Anthropic SSE events for an error response.

    Returns a properly formatted streaming response that includes usage data,
    so Claude Code doesn't crash trying to access undefined usage.input_tokens.
    """
    msg_id = f"msg_error_{hash(error_message) & 0xFFFFFFFF:08x}"
    error_text = f"[Proxy Error] {error_message}"
    output_tokens = len(error_text) // 4

    return [
        f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'usage': {'input_tokens': input_tokens, 'output_tokens': 0}}})}\n\n",
        f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n",
        f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': error_text}})}\n\n",
        f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n",
        f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'input_tokens': input_tokens, 'output_tokens': output_tokens}})}\n\n",
        f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n",
    ]


# ============ Modal Configuration ============

MODAL_APP_NAME = "sera-demo-vllm"
MODAL_VLLM_PORT = 8000
MODAL_MAX_MODEL_LEN = 32768
MODAL_GPU = "H100"
MODAL_VOLUME_NAME = "swe-agent-models"
MODAL_MODELS_DIR = "/models"
MODAL_HF_SECRET_NAME = "huggingface"

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


def _get_model_local_path() -> str:
    """Get the local path where the model should be stored in the volume."""
    model_name = _get_modal_model().split("/")[-1]
    return f"{MODAL_MODELS_DIR}/{model_name}"


def _ensure_model_downloaded() -> str:
    """Download model to volume if not already present. Returns local path."""
    from huggingface_hub import snapshot_download

    local_path = _get_model_local_path()

    # Check if model already exists
    if os.path.exists(local_path) and os.listdir(local_path):
        print(f"Model already cached at {local_path}")
        return local_path

    modal_model = _get_modal_model()
    print(f"Downloading {modal_model} to {local_path}...")
    print("(This only happens once - subsequent runs will use the cached model)")

    # Enable fast xet transfer for HuggingFace downloads
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # Use HF_TOKEN from environment if available (injected by Modal secret)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Using HuggingFace token from Modal secret")

    snapshot_download(
        repo_id=modal_model,
        local_dir=local_path,
        ignore_patterns=["*.md", "*.txt"],  # Skip docs to save space
        token=hf_token,  # Pass token for private/gated models
    )

    # Commit the volume so the download persists
    modal_volume.commit()
    print(f"Model downloaded and cached at {local_path}")
    return local_path


def modal_vllm_server():
    """Start vLLM server - Modal routes traffic to it."""
    # Ensure model is downloaded (uses cached version if available)
    model_path = _ensure_model_downloaded()

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

    # Use the HuggingFace model ID as the served name so users don't need to know about /models/ path
    cmd.extend(["--served-model-name", _get_modal_model()])

    # Add API key authentication if configured
    vllm_api_key = os.environ.get("VLLM_API_KEY")
    if vllm_api_key:
        cmd.extend(["--api-key", vllm_api_key])
    print(f"Starting vLLM with model: {model_path}")
    subprocess.Popen(cmd)


def register_modal_function(
    hf_secret: str | None = None, api_key: str | None = None
) -> None:
    """Register the Modal vLLM server function with optional HF secret and API key.

    Must be called before deploying the Modal app. The secret should contain
    an HF_TOKEN environment variable with a valid HuggingFace token.
    """
    # Pass the model name and API key to Modal via environment variable (CONFIG is local-only)
    env_dict: dict[str, str | None] = {"MODAL_MODEL": CONFIG.model}
    if api_key:
        env_dict["VLLM_API_KEY"] = api_key
    secrets = [modal.Secret.from_dict(env_dict)]
    if hf_secret:
        secrets.append(modal.Secret.from_name(hf_secret))

    # Apply decorators to register the Modal function
    modal_app.function(
        image=modal_image,
        gpu=MODAL_GPU,
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


SWE_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": """Custom editing tool for viewing, creating and editing files.
* State is persistent across command calls and discussions with the user.
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep.
* The `create` command cannot be used if the specified `path` already exists as a file.
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`.

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique.
* The `new_str` parameter should contain the edited lines that should replace the `old_str`.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`.",
                        "enum": ["view", "create", "str_replace"],
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`.",
                    },
                    "file_text": {
                        "type": "string",
                        "description": "Required parameter of `create` command, with the content of the file to be created.",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Required parameter of `str_replace` command containing the string in `path` to replace.",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Optional parameter of `str_replace` command containing the new string (if not given, no string will be added).",
                    },
                    "view_range": {
                        "type": "array",
                        "description": "Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.",
                        "items": {"type": "integer"},
                    },
                },
                "required": ["command", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit your solution when the task is complete.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


def strip_system_reminders(text: str) -> str:
    """Strip Claude Code system reminders from tool output.

    Claude Code injects <system-reminder>...</system-reminder> blocks
    into tool results, which are not part of the SWE-agent format.
    """
    return re.sub(
        r"\n?<system-reminder>.*?</system-reminder>\n?", "", text, flags=re.DOTALL
    )


def convert_line_format(cc_result: str) -> str:
    """Convert Claude Code line format (arrow) to SWE-agent format (tab).

    Claude Code Read returns lines like:
        '     1->content'
    SWE-agent expects:
        '     1\tcontent'
    """
    # Pattern: 6-char right-justified number followed by arrow
    # Replace arrow with tab character
    return re.sub(r"^(\s*\d+)\u2192", r"\1\t", cc_result, flags=re.MULTILINE)


def format_tool_result(tool_call_id: str, cc_result: str) -> str:
    """Format Claude Code tool result to match SWE-agent's expected output.

    Uses a cache to ensure historical messages get the same formatting as when
    they were first processed. This avoids issues with file state changes.
    """
    # Strip Claude Code system reminders first - they're not part of SWE-agent format
    cc_result = strip_system_reminders(cc_result)

    # Log the raw cc_result for debugging
    cc_preview = cc_result[:200].replace("\n", "\\n") if cc_result else "(empty)"
    log.debug(
        f"format_result: tool_id={tool_call_id[:16]}... cc_result=[{cc_preview}...]"
    )

    # IMPORTANT: Check pending_tool_calls FIRST to distinguish current vs historical
    # Current tool results need fresh file reads; cache should ONLY be used for historical
    ctx = pending_tool_calls.pop(tool_call_id, None)

    if not ctx:
        # No context - this is a historical message, use cache if available
        if tool_call_id in tool_result_cache:
            cached = tool_result_cache[tool_call_id]
            cached_preview = cached[:100].replace("\n", "\\n")
            log.debug(
                f"format_result: CACHE HIT (historical) for tool_id={tool_call_id[:16]}... returning=[{cached_preview}...]"
            )
            return cached
        # Not in cache either - format minimally and cache
        log.debug(
            f"format_result: no context for {tool_call_id[:16]}... (historical, not cached)"
        )
        formatted = convert_line_format(cc_result)
        tool_result_cache[tool_call_id] = formatted
        return formatted

    # Has context - this is a current tool result, process fresh (don't use cache)
    log.debug(
        f"format_result: has_context=True for tool_id={tool_call_id[:16]}... (current, processing fresh)"
    )

    swe_name = ctx["swe_name"]
    swe_input = ctx["swe_input"]
    formatted = cc_result  # Default to original

    if swe_name == "str_replace_editor":
        path = swe_input.get("path", "")  # Original path (for display to model)
        local_path = strip_testbed_prefix(path)  # Actual path (for file operations)
        cmd = swe_input.get("command")

        if cmd == "view":
            # Check if path is a directory - use local_path for actual file system check
            # This correctly handles ".", "..", and paths without extensions
            if os.path.isdir(local_path):
                # Directory listing result - matches SWE-agent's find output format
                # SWE-agent adds trailing newline: f"...\n{stdout}\n"
                log.debug(f"format_result: str_replace_editor view {path} (dir)")
                result = cc_result.rstrip("\n")  # Remove any trailing newlines
                formatted = f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{result}\n"
            else:
                # Convert Claude Code's line format (arrow) to SWE-agent's format (tab)
                # SWE-agent adds trailing newline after content
                converted_result = convert_line_format(cc_result).rstrip("\n")
                formatted = f"Here's the result of running `cat -n` on {path}:\n{converted_result}\n"

        elif cmd == "create":
            formatted = f"File created successfully at: {path}"

        elif cmd == "str_replace":
            old_str = swe_input.get("old_str", "")

            # Check if Claude Code returned an error - detect common error patterns
            # Claude Code Edit tool returns errors like "old_string was not found" or "not unique"
            cc_result_lower = cc_result.lower()
            is_error = (
                "error" in cc_result_lower
                or "not found" in cc_result_lower
                or "not unique" in cc_result_lower
                or "no match" in cc_result_lower
                or "multiple" in cc_result_lower
            )

            if is_error:
                # Convert Claude Code error to SWE-agent format
                log.debug(
                    f"format_result: str_replace error detected: {cc_result[:100]}"
                )
                if "not unique" in cc_result_lower or "multiple" in cc_result_lower:
                    formatted = (
                        f"No replacement was performed. Multiple occurrences of old_str "
                        f"`{old_str[:100]}` in the file. Please ensure it is unique."
                    )
                else:
                    # Default to "not found" error format
                    formatted = (
                        f"No replacement was performed, old_str `{old_str[:100]}` "
                        f"did not appear verbatim in {path}."
                    )
            else:
                # Success case - read the file and generate a snippet like SWE-agent does
                # Since proxy runs in same CWD as Claude Code, we can read the file directly
                try:
                    new_str = swe_input.get("new_str", "")
                    with open(local_path, "r", encoding="utf-8", errors="replace") as f:
                        file_content = f.read()

                    # Find where the new content is located
                    # Calculate the line number where replacement happened
                    lines = file_content.split("\n")
                    total_lines = len(lines)

                    # Find the line containing new_str (or first line of new_str if multiline)
                    new_str_first_line = new_str.split("\n")[0] if new_str else ""
                    replacement_line = 0
                    for i, line in enumerate(lines):
                        if new_str_first_line and new_str_first_line in line:
                            replacement_line = i
                            break
                        elif new_str and new_str in "\n".join(
                            lines[i : i + new_str.count("\n") + 1]
                        ):
                            replacement_line = i
                            break

                    # Extract snippet with 4 lines before and after (SWE-agent's SNIPPET_LINES = 4)
                    snippet_lines = 4
                    start_line = max(0, replacement_line - snippet_lines)
                    end_line = min(
                        replacement_line + snippet_lines + new_str.count("\n") + 1,
                        total_lines,
                    )
                    snippet = "\n".join(lines[start_line:end_line])

                    # Format with line numbers (1-indexed, 6-char width, tab separator)
                    numbered_lines = []
                    for i, line in enumerate(snippet.split("\n")):
                        line_num = start_line + i + 1  # 1-indexed
                        numbered_lines.append(f"{line_num:6}\t{line}")
                    formatted_snippet = "\n".join(numbered_lines)

                    formatted = (
                        f"The file {path} has been edited. "
                        f"Here's the result of running `cat -n` on a snippet of {path}:\n"
                        f"{formatted_snippet}\n"
                        "Review the changes and make sure they are as expected. Edit the file again if necessary."
                    )
                except Exception as e:
                    # Fall back to simple message if file read fails
                    log.warning(f"Failed to generate snippet for {local_path}: {e}")
                    formatted = (
                        f"The file {path} has been edited. "
                        "Review the changes and make sure they are as expected. Edit the file again if necessary."
                    )

    elif swe_name == "bash":
        # SWE-agent returns empty string for no output, not "(no output)"
        formatted = cc_result

    # Cache the formatted result for historical message handling (keyed by unique tool_call_id)
    tool_result_cache[tool_call_id] = formatted
    formatted_preview = formatted[:100].replace("\n", "\\n")
    log.debug(
        f"format_result: CACHED new result for tool_id={tool_call_id[:16]}... result=[{formatted_preview}...]"
    )
    return formatted


def reverse_map_tool_call(name: str, input_args: dict) -> tuple[str, dict]:
    """Map Claude Code tool call back to SWE-agent format for historical messages.

    This ensures the model sees consistent SWE-agent tool names in conversation history.
    """
    if name == "Bash":
        command = input_args.get("command", "")
        # Detect directory listing pattern that was converted from str_replace_editor view
        # Pattern: find "{path}" -maxdepth 2 -not -path "*/.*" | head -100
        match = re.match(
            r'^find "([^"]+)" -maxdepth 2 -not -path "\*/\.\*" \| head -100$', command
        )
        if match:
            # This was a str_replace_editor view on a directory - reverse map it
            path = match.group(1)
            log.debug(
                f"reverse_map: Bash(find {path}) -> str_replace_editor view {path}"
            )
            return "str_replace_editor", {"command": "view", "path": path}
        return "bash", input_args

    if name == "Read":
        # Convert Read -> str_replace_editor view
        swe_input = {
            "command": "view",
            "path": input_args.get("file_path", ""),
        }
        # Convert offset/limit back to view_range if present
        if "offset" in input_args and "limit" in input_args:
            start = input_args["offset"]
            end = start + input_args["limit"] - 1
            swe_input["view_range"] = [start, end]
        return "str_replace_editor", swe_input

    if name == "Write":
        # Convert Write -> str_replace_editor create
        return "str_replace_editor", {
            "command": "create",
            "path": input_args.get("file_path", ""),
            "file_text": input_args.get("content", ""),
        }

    if name == "Edit":
        # Convert Edit -> str_replace_editor str_replace
        return "str_replace_editor", {
            "command": "str_replace",
            "path": input_args.get("file_path", ""),
            "old_str": input_args.get("old_string", ""),
            "new_str": input_args.get("new_string", ""),
        }

    # Unknown tool - pass through unchanged
    return name, input_args


def convert_message(msg: dict) -> dict | list[dict]:
    """Convert Anthropic message format to OpenAI format."""
    role = msg.get("role")
    content = msg.get("content")

    # Simple string content
    if isinstance(content, str):
        # Wrap assistant text in <think> tags for model consistency
        if role == "assistant" and content:
            content = wrap_in_think_tags(content)
        return {"role": role, "content": content}

    # Array of content blocks (Anthropic style)
    if isinstance(content, list):
        # Handle tool_result blocks -> OpenAI tool response
        tool_results = [c for c in content if c.get("type") == "tool_result"]
        if tool_results:
            # OpenAI expects separate messages for each tool result
            return [
                {
                    "role": "tool",
                    "tool_call_id": tr["tool_use_id"],
                    "content": format_tool_result(
                        tr["tool_use_id"],
                        tr["content"]
                        if isinstance(tr["content"], str)
                        else json.dumps(tr["content"]),
                    ),
                }
                for tr in tool_results
            ]

        # Handle tool_use blocks (assistant messages with tool calls)
        # Reverse-map Claude Code tools back to SWE-agent format for model consistency
        tool_uses = [c for c in content if c.get("type") == "tool_use"]
        if tool_uses:
            text_content = next(
                (c["text"] for c in content if c.get("type") == "text"), ""
            )
            # Wrap assistant text in <think> tags for model consistency
            if text_content:
                text_content = wrap_in_think_tags(text_content)
            tool_calls = []
            for tu in tool_uses:
                swe_name, swe_input = reverse_map_tool_call(tu["name"], tu["input"])
                tool_calls.append(
                    {
                        "id": tu["id"],
                        "type": "function",
                        "function": {
                            "name": swe_name,
                            "arguments": json.dumps(swe_input),
                        },
                    }
                )
            return {
                "role": "assistant",
                "content": text_content,
                "tool_calls": tool_calls,
            }

        # Just text blocks - wrap assistant content in <think> tags
        text = "\n".join(c.get("text", "") for c in content if c.get("type") == "text")
        if role == "assistant" and text:
            text = wrap_in_think_tags(text)
        return {"role": role, "content": text}

    return {"role": role, "content": str(content)}


def map_tool_call(name: str, input_args: dict, tool_id: str) -> dict | None:
    """Map SWE-agent tool call to Claude Code tool, storing context for result formatting.

    Returns None for 'submit' tool to signal task completion (triggers end_turn).
    """
    # Handle submit tool first - signals task completion, don't pass to Claude Code
    if name == "submit":
        log.debug("map_tool: submit -> end_turn")
        return None

    # Store context for result formatting later
    pending_tool_calls[tool_id] = {"swe_name": name, "swe_input": input_args}

    if name == "bash":
        command = strip_testbed_from_command(input_args.get("command", ""))
        return {
            "type": "tool_use",
            "id": tool_id,
            "name": "Bash",
            "input": {"command": command},
        }

    if name == "str_replace_editor":
        cmd = input_args.get("command")
        path = strip_testbed_prefix(input_args.get("path", ""))

        if cmd == "view":
            view_range = input_args.get("view_range")
            # Check if path is a directory - use os.path.isdir since proxy runs in same CWD
            # This correctly handles ".", "..", and paths without extensions
            if os.path.isdir(path):
                # Directory view - list files up to 2 levels deep (SWE-agent behavior)
                log.debug(
                    f"map_tool: str_replace_editor view {path} (dir) -> Bash find"
                )
                return {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": "Bash",
                    "input": {
                        "command": f'find "{path}" -maxdepth 2 -not -path "*/.*" | head -100'
                    },
                }
            # File view
            log.debug(f"map_tool: str_replace_editor view {path} -> Read")
            cc_input: dict[str, Any] = {"file_path": path}
            if view_range and len(view_range) >= 2:
                cc_input["offset"] = view_range[0]
                cc_input["limit"] = view_range[1] - view_range[0] + 1
            return {
                "type": "tool_use",
                "id": tool_id,
                "name": "Read",
                "input": cc_input,
            }

        if cmd == "create":
            return {
                "type": "tool_use",
                "id": tool_id,
                "name": "Write",
                "input": {
                    "file_path": path,
                    "content": input_args.get("file_text", ""),
                },
            }

        if cmd == "str_replace":
            return {
                "type": "tool_use",
                "id": tool_id,
                "name": "Edit",
                "input": {
                    "file_path": path,
                    "old_string": input_args.get("old_str", ""),
                    "new_string": input_args.get("new_str", ""),
                },
            }

    # Unknown tool - pass through (hybrid support)
    return {"type": "tool_use", "id": tool_id, "name": name, "input": input_args}


def strip_testbed_prefix(path: str) -> str:
    """Strip /testbed/ prefix from paths.

    The model was trained with /testbed/ as the working directory,
    but we need paths relative to the actual CWD.
    """
    if path.startswith("/testbed/"):
        return path[9:]  # len("/testbed/") == 9
    if path == "/testbed":
        return "."
    return path


def strip_testbed_from_command(command: str) -> str:
    """Strip /testbed/ references from bash commands.

    Only replaces /testbed when it appears at the start of a path, i.e.:
    - At start of string
    - After whitespace
    - After quotes (" or ')
    - After = (variable assignments)

    This avoids breaking paths like /home/testbed/ or my_project/testbed/
    """
    # Pattern matches /testbed/ or /testbed at path boundaries
    # Lookbehind for start of string, whitespace, quotes, or =
    # Replace /testbed/ with ./ and standalone /testbed with .
    result = re.sub(r'(^|(?<=\s)|(?<=["\'=]))/testbed/', r"\1./", command)
    result = re.sub(r'(^|(?<=\s)|(?<=["\'=]))/testbed(?=\s|$|["\'"])', r"\1.", result)
    return result


def strip_think_tags(text: str) -> str:
    """Remove <think> and </think> tags from text, keeping the content."""
    text = re.sub(r"<think>\n?", "", text)
    text = re.sub(r"\n?</think>\n?", "", text)
    return text.strip()


def wrap_in_think_tags(text: str) -> str:
    """Wrap text in <think> tags if not already wrapped."""
    text = text.strip()
    if not text:
        return text
    if text.startswith("<think>"):
        return text
    return f"<think>\n{text}\n</think>"


async def stream_anthropic_response(
    openai_stream: AsyncIterator[str],
    model: str,
) -> AsyncIterator[str]:
    """Convert OpenAI streaming response to Anthropic streaming format."""
    msg_id = f"msg_{id(openai_stream)}"
    current_tool_calls: dict[int, dict] = {}  # index -> {id, name, arguments_str}
    accumulated_text = ""  # Buffer text to strip think tags at the end
    input_tokens = 0
    output_tokens = 0

    # Send message_start event
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

    async for line in openai_stream:
        line_str = line.strip()
        if not line_str or not line_str.startswith("data: "):
            continue

        data_str = line_str[6:]  # Remove "data: " prefix
        if data_str == "[DONE]":
            break

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        # Track usage if present
        if "usage" in chunk:
            usage = chunk["usage"]
            input_tokens = usage.get("prompt_tokens", input_tokens)
            output_tokens = usage.get("completion_tokens", output_tokens)

        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Accumulate text content (will be emitted at end with think tags stripped)
        if "content" in delta and delta["content"]:
            accumulated_text += delta["content"]

        # Handle tool calls
        if "tool_calls" in delta:
            for tc in delta["tool_calls"]:
                tc_index = tc.get("index", 0)
                if tc_index not in current_tool_calls:
                    # New tool call starting
                    current_tool_calls[tc_index] = {
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments_str": "",
                    }

                # Accumulate function arguments
                if "function" in tc:
                    if "name" in tc["function"] and tc["function"]["name"]:
                        current_tool_calls[tc_index]["name"] = tc["function"]["name"]
                    if "arguments" in tc["function"]:
                        current_tool_calls[tc_index]["arguments_str"] += tc["function"][
                            "arguments"
                        ]
                if "id" in tc and tc["id"]:
                    current_tool_calls[tc_index]["id"] = tc["id"]

        # Handle finish
        if finish_reason:
            # Emit accumulated text with think tags stripped
            cleaned_text = strip_think_tags(accumulated_text)
            has_text = bool(cleaned_text)

            if has_text:
                # Send text content block
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': cleaned_text}})}\n\n"
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

            # Now emit tool_use blocks
            block_index = 1 if has_text else 0
            emitted_tool_use = False
            for tc_idx in sorted(current_tool_calls.keys()):
                tc_data = current_tool_calls[tc_idx]
                try:
                    input_args = (
                        json.loads(tc_data["arguments_str"])
                        if tc_data["arguments_str"]
                        else {}
                    )
                except json.JSONDecodeError:
                    input_args = {}

                # Map to Claude Code tool
                mapped = map_tool_call(tc_data["name"], input_args, tc_data["id"])

                # Skip submit tool (returns None) - signals task completion
                if mapped is None:
                    continue

                emitted_tool_use = True

                # Send content_block_start for tool_use
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': mapped['id'], 'name': mapped['name'], 'input': {}}})}\n\n"

                # Send input_json_delta with the full input
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(mapped['input'])}})}\n\n"

                # Close the tool_use block
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                block_index += 1

            # Log streaming response summary
            tool_names = [
                current_tool_calls[i]["name"] for i in sorted(current_tool_calls.keys())
            ]
            text_preview = cleaned_text[:100].replace("\n", " ") if cleaned_text else ""
            log.debug(
                f"<- vLLM (stream): tools={tool_names} content=[{text_preview}...]"
            )

            # Determine stop reason - only tool_use if we emitted actual tool calls
            stop_reason = "tool_use" if emitted_tool_use else "end_turn"

            # Send message_delta with stop_reason
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason}, 'usage': {'output_tokens': output_tokens}})}\n\n"

    # Send message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


def get_vllm_request_headers() -> dict[str, str]:
    """Get headers for requests to vLLM endpoint, including auth if configured."""
    headers = {"Content-Type": "application/json"}
    if CONFIG.api_key:
        headers["Authorization"] = f"Bearer {CONFIG.api_key}"
    return headers


def create_app():
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="SWE-agent Proxy", description="Anthropic <-> OpenAI translation proxy"
    )

    @app.post("/api/event_logging/batch")
    async def event_logging(request: Request):
        """Dummy endpoint for Claude Code telemetry (ignored)."""
        return JSONResponse(content={"status": "ok"})

    @app.post("/v1/messages")
    async def proxy_messages(request: Request):
        """Proxy Anthropic API requests to vLLM with format translation."""
        try:
            anthropic_req = await request.json()

            # Convert messages, flattening any lists (from tool_result conversion)
            openai_messages = []

            # Prepend system prompt if configured
            if CONFIG.system_prompt:
                openai_messages.append(
                    {"role": "system", "content": CONFIG.system_prompt}
                )

            for msg in anthropic_req.get("messages", []):
                converted = convert_message(msg)
                if isinstance(converted, list):
                    openai_messages.extend(converted)
                else:
                    openai_messages.append(converted)

            requested_max_tokens = min(anthropic_req.get("max_tokens", 4096), 8192)

            openai_req = {
                "model": CONFIG.model,
                "messages": openai_messages,
                # Tools array - vLLM's Hermes chat template injects these into
                # the prompt as <tools> XML and adds <tool_call> instructions
                "tools": SWE_AGENT_TOOLS,
                "max_tokens": requested_max_tokens,
                "temperature": anthropic_req.get("temperature", 0),
                "stream": True,
            }

            # Log request summary (not full JSON with repeated tool definitions)
            msg_summary = []
            for m in openai_messages:
                role = m.get("role", "?")
                if role == "assistant" and "tool_calls" in m:
                    calls = [
                        tc.get("function", {}).get("name", "?")
                        for tc in m["tool_calls"]
                    ]
                    msg_summary.append(f"assistant[tool:{','.join(calls)}]")
                elif role == "tool":
                    msg_summary.append(f"tool[{m.get('tool_call_id', '?')[:8]}]")
                else:
                    content_preview = str(m.get("content", ""))[:50].replace("\n", " ")
                    msg_summary.append(f"{role}[{content_preview}...]")
            log.debug(
                f"-> vLLM: {len(openai_messages)} msgs: {' | '.join(msg_summary)}"
            )

            async def generate_stream():
                current_max_tokens = openai_req["max_tokens"]

                async with httpx.AsyncClient(timeout=300) as client:
                    # First attempt
                    async with client.stream(
                        "POST",
                        CONFIG.endpoint,
                        json={**openai_req, "max_tokens": current_max_tokens},
                        headers=get_vllm_request_headers(),
                    ) as response:
                        if response.status_code != 200:
                            error_bytes = await response.aread()
                            error_text = error_bytes.decode()
                            log.error(f"vLLM error: {error_text}")

                            # Check if this is a context overflow error we can retry
                            overflow = parse_context_overflow_error(error_text)
                            if overflow:
                                input_tokens, max_context = overflow

                                # If input alone exceeds context, we can't retry
                                if input_tokens >= max_context:
                                    log.error(
                                        f"Context exhausted: {input_tokens} input tokens "
                                        f">= {max_context} max context"
                                    )
                                    for event in generate_error_sse_events(
                                        f"Context exhausted: {input_tokens} input tokens exceed "
                                        f"model's {max_context} token limit. Please start a new conversation.",
                                        CONFIG.model,
                                        input_tokens,
                                    ):
                                        yield event
                                    return

                                # Calculate max_tokens that fits, with 100 token buffer
                                new_max_tokens = max(
                                    256, max_context - input_tokens - 100
                                )
                                log.info(
                                    f"Context overflow: {input_tokens} input tokens, "
                                    f"retrying with max_tokens={new_max_tokens}"
                                )

                                # Retry with reduced max_tokens
                                async with client.stream(
                                    "POST",
                                    CONFIG.endpoint,
                                    json={**openai_req, "max_tokens": new_max_tokens},
                                    headers=get_vllm_request_headers(),
                                ) as retry_response:
                                    if retry_response.status_code != 200:
                                        retry_error = await retry_response.aread()
                                        log.error(f"vLLM retry error: {retry_error}")
                                        # Return error with usage data
                                        for event in generate_error_sse_events(
                                            retry_error.decode(),
                                            CONFIG.model,
                                            input_tokens,
                                        ):
                                            yield event
                                        return

                                    async for chunk in stream_anthropic_response(
                                        retry_response.aiter_lines(),
                                        CONFIG.model,
                                    ):
                                        yield chunk
                                return

                            # Not a context overflow error, return error with usage=0
                            for event in generate_error_sse_events(
                                error_text, CONFIG.model, 0
                            ):
                                yield event
                            return

                        async for chunk in stream_anthropic_response(
                            response.aiter_lines(),
                            CONFIG.model,
                        ):
                            yield chunk

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        except Exception as e:
            log.error(f"Proxy error: {e}\n{traceback.format_exc()}")
            return JSONResponse(
                status_code=500,
                content={
                    "type": "error",
                    "error": {"type": "api_error", "message": str(e)},
                },
            )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "vllm_endpoint": CONFIG.endpoint,
            "model": CONFIG.model,
        }

    return app


def start_proxy_background(port: int) -> None:
    """Start the proxy server in a background thread."""
    app = create_app()
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    time.sleep(1)


def print_proxy_status(port: int) -> None:
    """Print proxy status information."""
    print(f"Proxy running on http://localhost:{port}")
    print(f"Forwarding to vLLM: {CONFIG.endpoint}")
    print(f"Model: {CONFIG.model}")
    print(f"Logs: {LOG_FILE}")
    print()


def run_claude_code(port: int) -> int:
    """Run Claude Code with proxy env var. Returns exit code."""
    print(f"Running: ANTHROPIC_BASE_URL=http://localhost:{port} claude")
    print("=" * 60)
    print()

    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = f"http://localhost:{port}"
    proc = subprocess.run(["claude"], env=env)
    return proc.returncode


def print_claude_code_hint(port: int) -> None:
    """Print hint for running Claude Code manually."""
    print("To run Claude Code:")
    print(f"  ANTHROPIC_BASE_URL=http://localhost:{port} claude")
    print()


def wait_for_vllm_ready(base_url: str, timeout: int = 1200) -> bool:
    """Wait for vLLM server to be ready by polling /v1/models endpoint.

    Args:
        base_url: The base URL of the vLLM server (e.g., https://xxx.modal.run)
        timeout: Maximum time to wait in seconds (default 20 minutes)

    Returns:
        True if server is ready, False if timeout exceeded
    """
    models_url = f"{base_url}/v1/models"
    start_time = time.time()

    print("Waiting for vLLM to be ready", end="", flush=True)

    while time.time() - start_time < timeout:
        try:
            # Include API key header if configured (vLLM protects /v1/* routes)
            headers = get_vllm_request_headers()
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
    """Stop the Modal app and print status."""
    print()
    print("Stopping Modal vLLM server...")
    subprocess.run(
        ["modal", "app", "stop", MODAL_APP_NAME],
        capture_output=True,
    )
    print("Modal server stopped.")


def deploy_modal_app(hf_secret: str | None = None) -> None:
    """Deploy the Modal app. Raises on failure.

    Args:
        hf_secret: Optional name of Modal secret containing HF_TOKEN for private models.
    """
    # Generate API key for this deployment
    api_key = str(uuid.uuid4())
    CONFIG.api_key = api_key

    # Register the Modal function with optional HF secret and API key before deploying
    register_modal_function(hf_secret=hf_secret, api_key=api_key)

    print("Deploying vLLM to Modal...")
    if hf_secret:
        print(f"Using Modal secret '{hf_secret}' for HuggingFace authentication")
    print("(This may take a few minutes on first run while the container builds)")
    print()

    try:
        with modal.enable_output():
            modal_app.deploy()
    except Exception as e:
        raise RuntimeError(f"Failed to deploy Modal app: {e}") from e


def configure_modal_endpoint() -> None:
    """Configure the proxy to use the deployed Modal endpoint."""
    # Get the endpoint URL
    vllm_fn = modal.Function.from_name(MODAL_APP_NAME, "modal_vllm_server")
    vllm_url = vllm_fn.get_web_url()
    if vllm_url is None:
        raise RuntimeError("Could not get vLLM endpoint URL from Modal.")

    # Wait for vLLM to be fully ready (model loaded)
    print()
    if not wait_for_vllm_ready(vllm_url):
        raise RuntimeError("vLLM server failed to become ready")

    # Configure proxy to use Modal endpoint
    CONFIG.endpoint = f"{vllm_url}/v1/chat/completions"
    # CONFIG.model already set - vLLM serves it via --served-model-name


def run(
    port: int,
    modal: bool = False,
    proxy_only: bool = False,
    hf_secret: str | None = None,
) -> None:
    """Run the proxy, optionally deploying to Modal and/or launching Claude Code.

    Args:
        port: Port for the proxy server.
        modal: If True, deploy vLLM to Modal.
        proxy_only: If True, don't launch Claude Code.
        hf_secret: Optional name of Modal secret containing HF_TOKEN.
    """
    exit_code = 0
    modal_deployed = False
    try:
        # Step 1: Set up vLLM endpoint (deploy Modal if needed)
        if modal:
            deploy_modal_app(hf_secret=hf_secret)
            modal_deployed = True
            configure_modal_endpoint()

        # Step 2: Start proxy
        start_proxy_background(port)
        print()
        print_proxy_status(port)

        # Step 3: Either launch Claude Code or block
        if proxy_only:
            print_claude_code_hint(port)
            if modal:
                print("Press Ctrl+C to stop the server and shut down Modal.")
            print("=" * 60)
            while True:
                time.sleep(1)
        else:
            exit_code = run_claude_code(port)
    except KeyboardInterrupt:
        print("\nExiting...")
    except FileNotFoundError:
        print("Error: 'claude' command not found. Is Claude Code installed?")
        print("Install with: npm install -g @anthropic-ai/claude-code")
        exit_code = 1
    except RuntimeError as e:
        print(f"Error: {e}")
        exit_code = 1
    finally:
        # Step 4: Cleanup (Modal only)
        if modal_deployed:
            stop_modal_app()

    sys.exit(exit_code)


def main():
    parser = argparse.ArgumentParser(
        description="SWE-agent <-> Claude Code translation proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    sera --endpoint URL              # Start proxy and launch Claude Code
    sera --endpoint URL --proxy-only # Start proxy server only
    sera --modal                     # Deploy to Modal and launch Claude Code
    sera --modal --proxy-only        # Deploy to Modal (proxy only)
    sera --modal --hf-secret mysecret --model org/private-model  # Use private model

To use private HuggingFace models with Modal:
    1. Create a Modal secret: modal secret create huggingface HF_TOKEN=hf_xxxxx
    2. Run with: sera --modal --hf-secret huggingface --model your-org/model
        """,
    )
    parser.add_argument(
        "--proxy-only",
        action="store_true",
        help="Start proxy server only (don't launch Claude Code)",
    )
    parser.add_argument(
        "--modal",
        action="store_true",
        help="Deploy vLLM to Modal instead of using local/remote endpoint",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=CONFIG.port,
        help=f"Port for proxy server (default: {CONFIG.port})",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="vLLM endpoint URL (required unless --modal is used)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name/path (HuggingFace ID for --modal, vLLM model path otherwise)",
    )
    parser.add_argument(
        "--hf-secret",
        default=None,
        help=(
            "Modal secret name containing HF_TOKEN for private/gated models. "
            "Create with: modal secret create <name> HF_TOKEN=<your-token>"
        ),
    )

    args = parser.parse_args()

    # Environment variable fallbacks (CLI args > env vars > defaults)
    if args.model is None:
        args.model = os.environ.get("SERA_MODEL")
    if args.hf_secret is None:
        args.hf_secret = os.environ.get("SERA_HF_SECRET")

    # Validate: --endpoint is required unless --modal is used
    if not args.modal and not args.endpoint:
        parser.error("--endpoint is required unless --modal is used")

    # Update config from CLI args
    CONFIG.port = args.port
    if args.model:
        CONFIG.model = args.model
    if args.modal:
        # API key is auto-generated in deploy_modal_app()
        pass
    else:
        CONFIG.endpoint = args.endpoint
        # Check for SERA_API_KEY environment variable for direct endpoint auth
        api_key = os.environ.get("SERA_API_KEY")
        if api_key:
            CONFIG.api_key = api_key

    run(
        args.port,
        modal=args.modal,
        proxy_only=args.proxy_only,
        hf_secret=args.hf_secret,
    )


if __name__ == "__main__":
    main()
