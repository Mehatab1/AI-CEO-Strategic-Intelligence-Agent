#!/bin/bash
# Re-run this at the start of any new data-lab session if Ollama/the model
# don't persist across container restarts (common on JupyterHub-style labs).
set -e

OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"

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