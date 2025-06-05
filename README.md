# AI Commit Message Generator

A CLI tool that generates commit messages using AI (Ollama) for the last commit in your current branch.

## Prerequisites

1. Python 3.7+
2. Ollama installed and running (https://ollama.ai)
3. Git repository with at least one commit

## Installation

1. Clone this repository
2. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Make sure Ollama is running and you have pulled your desired model (default is llama2):
```bash
ollama pull llama2
```

To generate a commit message for the last commit:
```bash
python core.py generate
```

To use a different model:
```bash
python core.py generate --model codellama
```

## Features

- Generates conventional commit messages based on the last commit's diff
- Uses local LLM through Ollama for privacy and speed
- Supports different Ollama models (default: llama2)
- Rich terminal output with status indicators 