#!/bin/bash

# FreeDocGPT One-Click Startup Script
set -e

echo "🚀 Starting FreeDocGPT..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Install with: brew install ollama"
    exit 1
fi

# Start Ollama if not running
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "📡 Starting Ollama..."
    ollama serve &
    sleep 3
fi

# Pull models
echo "🔄 Checking models..."
ollama pull embeddinggemma:300m &
ollama pull gpt-oss:20b &
wait

# Setup Python env
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install deps once
if [ ! -f ".venv/.installed" ]; then
    echo "📋 Installing dependencies..."
    pip install -r requirements.txt
    touch .venv/.installed
fi

# Create docs folder
mkdir -p documents

# Start app
echo "🎉 Starting app at http://localhost:8501"
streamlit run app.py