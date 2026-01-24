# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a translation proxy that enables Claude Code CLI to work with models fine-tuned on SWE-agent format (from swe-bench). It translates between:
- **Anthropic API format** (what Claude Code sends) ↔ **OpenAI API format** (what vLLM serves)
- **Claude Code tools** (Read, Edit, Write, Bash) ↔ **SWE-agent tools** (str_replace_editor, bash)

## Running

```bash
# Use Modal to host the model and launch Claude Code
sera --modal

# Or connect to an existing endpoint
sera --endpoint URL

# Deploy a persistent vLLM instance (for multi-user setups)
deploy-sera --model Qwen/Qwen3-32B
```

The proxy runs on port 8080, vLLM on port 8000.

## Project Structure

```
src/sera/
├── __init__.py   # Package entry point, exports main()
├── main.py       # Proxy server and CLI
└── deploy.py     # Standalone Modal deployment tool
```

CLI entry points:
- `sera` - Main proxy CLI (ephemeral Modal deployment)
- `deploy-sera` - Persistent Modal deployment for multi-user setups

## Building and Publishing

This project uses `uv_build` as its build backend.

```bash
# Build the package
just build

# Build for publishing (recommended)
just build-publish

# Dry run (build only, no publish)
just dry-run

# Publish to PyPI
PYPI_TOKEN=... just publish

# Publish to TestPyPI
TEST_PYPI_TOKEN=... just publish-test
```

## Architecture

**Request flow:**
```
Claude Code → sera (port 8080) → vLLM (port 8000)
     ↑              ↓                      ↓
Anthropic      Convert to OpenAI     SWE-agent tools
  format       format + map tools    (str_replace_editor, bash)
```

**Tool mapping in `src/sera/main.py`:**
- `str_replace_editor view` → `Read`
- `str_replace_editor create` → `Write`
- `str_replace_editor str_replace` → `Edit`
- `bash` → `Bash`

Results are reformatted to match SWE-agent's expected output style.

## Tool Calling Format (Hermes)

The model uses **Hermes-style function calling**:

1. **Proxy sends** a `tools` array with tool definitions to vLLM
2. **vLLM's Hermes chat template** injects tools into the prompt as `<tools>` XML and adds `<tool_call>` usage instructions
3. **Model outputs** tool calls as XML: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
4. **vLLM parses** these tags (via `--tool-call-parser hermes`) and converts them to structured `tool_calls` in the response

The system prompt is kept minimal ("You are a helpful assistant...") since vLLM handles tool injection. The `<think>` tags in model output are reasoning; actual tool invocations are in `<tool_call>` tags.

## Configuration

Configuration priority (highest to lowest):
1. CLI arguments
2. Environment variables
3. Defaults in `CONFIG`

**Environment variables:**
- `SERA_MODEL` - Model name/path (fallback for `--model`)
- `SERA_HF_SECRET` - Modal secret name for HuggingFace token (fallback for `--hf-secret`)
- `SERA_API_KEY` - API key for direct endpoint authentication

Run `sera --help` for CLI options.
