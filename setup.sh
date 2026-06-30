#!/bin/bash
# Re-run this at the start of any new data-lab session if Ollama/the model
# don't persist across container restarts (common on JupyterHub-style labs).
set -e

# Fully-autonomous mode (LLM workflow routing, free tool choice over all 7 tools, and the
# Stage 3b reflect/re-plan loop) needs a CAPABLE model. Small models (llama3.1:8b) are
# unreliable at unconstrained tool-calling and will frequently call no tool.
# Default to a strong open-source model; override with OLLAMA_MODEL if you have a bigger one.
#   recommended: qwen2.5:14b (balanced) · stronger: qwen2.5:32b / llama3.1:70b
#   minimal/legacy (not recommended for full autonomy): llama3.1:8b
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:14b}"

echo "=== Checking Ollama install ==="
if command -v ollama &> /dev/null; then
    echo "Ollama already installed, skipping install step."
else
    echo "Installing Ollama (requires zstd)..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq zstd
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo ""
echo "=== Starting Ollama server ==="
if pgrep -x "ollama" > /dev/null; then
    echo "Ollama server already running."
else
    ollama serve > /tmp/ollama.log 2>&1 &
    echo "Started ollama serve (PID $!), logging to /tmp/ollama.log"
    sleep 5
fi

echo ""
echo "=== Pulling model: $OLLAMA_MODEL ==="
ollama pull "$OLLAMA_MODEL"

echo ""
echo "=== Done. Installed models: ==="
ollama list