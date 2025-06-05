# Sayless - AI-Powered Git Commit Messages

A smart CLI tool that automatically generates meaningful commit messages using AI. It analyzes your staged changes and generates conventional commit messages that are clear, concise, and follow best practices.

## Features

- 🤖 Uses OpenAI's GPT models by default (with Ollama as fallback)
- 🎯 Generates conventional commit messages (feat, fix, docs, etc.)
- 🔄 Supports both cloud (OpenAI) and local (Ollama) AI
- 🚀 Easy setup and configuration
- 💡 Smart fallback between providers
- 🎨 Beautiful terminal output

## Prerequisites

1. Python 3.7+
2. Git repository with staged changes
3. OpenAI API key (recommended) or Ollama (alternative)

## Installation

1. Clone this repository
2. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Setup

### Option 1: OpenAI (Recommended)

Set up your OpenAI API key using one of these methods:

```bash
# Method 1: Environment variable
export OPENAI_API_KEY=your_api_key

# Method 2: CLI configuration
sayless config --openai-key your_api_key
```

### Option 2: Ollama (Local AI)

If you prefer to use local AI:

1. Install Ollama from https://ollama.ai
2. Switch to Ollama:
```bash
sayless config --use-ollama
```

## Usage

1. Stage your changes:
```bash
git add .  # or git add <specific-files>
```

2. Generate and create commit:
```bash
sayless generate
```

3. Preview without committing:
```bash
sayless generate --preview
```

## Configuration

View current settings:
```bash
sayless config --show
```

### AI Provider

```bash
# Use OpenAI (default)
sayless config --use-openai

# Use Ollama (local AI)
sayless config --use-ollama
```

### Models

```bash
# OpenAI models
sayless config --model gpt-4
sayless config --model gpt-4o  # default

# Ollama models
sayless config --model llama2  # default for Ollama
sayless config --model codellama
```

### OpenAI API Key

```bash
# Set API key
sayless config --openai-key YOUR_API_KEY

# Using environment variable
export OPENAI_API_KEY=your_api_key
```

## How It Works

1. Checks for staged changes in your git repository
2. Analyzes the diff using AI (OpenAI by default, Ollama as fallback)
3. Generates a conventional commit message
4. Creates the commit after your confirmation

## Conventional Commits

Generated messages follow the conventional commits format:

```
<type>(<scope>): <description>
```

Types:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding/fixing tests
- `chore`: Maintenance tasks

## Configuration File

Settings are stored in `~/.sayless/config.json`:
- AI provider (openai/ollama)
- OpenAI API key
- Default model

## Error Handling

- Automatically falls back to Ollama if OpenAI fails
- Clear error messages and setup instructions
- Validates git repository and staged changes
- Confirms commit messages before applying

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 