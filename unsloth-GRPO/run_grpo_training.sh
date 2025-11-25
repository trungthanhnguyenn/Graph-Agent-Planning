#!/bin/bash
# Script for Unsloth GRPO Training

echo "Starting Unsloth GRPO Training Script..."

# IMPORTANT: Set environment variables for Tool Executor
# Your reward function may need to call tools to evaluate the model's output.
# Based on the test_simple_integration.py file, there is an environment variable WIKI_RAG_SERVER_URL.
# Make sure you export it here or in your ~/.bashrc file
# For example:
# export WIKI_RAG_SERVER_URL="http://localhost:8008/retrieve"

# Check if the environment variable is set
if [ -z "$WIKI_RAG_SERVER_URL" ]; then
    echo "WARNING: Environment variable WIKI_RAG_SERVER_URL is not set."
    echo "Training may fail if the reward function needs to use the tool."
fi

# Get absolute path to the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Run Python training script
# Use python3 to ensure compatibility
python3 "$SCRIPT_DIR/train_grpo.py"

echo "Training completed."
