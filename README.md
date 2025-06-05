# Sayless 🤖

Sayless is an AI-powered Git commit message generator and repository analysis tool. It uses advanced AI models to generate meaningful commit messages, search through your commit history, and provide insightful summaries of your repository changes.

## Features

### 🎯 Smart Commit Message Generation
- Automatically generates meaningful commit messages based on your staged changes
- Supports both OpenAI (GPT-4) and Ollama (local AI) for message generation
- Preview mode to review generated messages before committing

### 🔍 Semantic Commit Search
- Search through your commit history using natural language
- AI-powered relevance scoring and grouping
- Automatic commit indexing for fast searches
- Detailed summaries of matching commits

### 📊 Repository Analysis
- Generate summaries of changes over specific time periods
- Detailed analysis of individual commits
- Impact assessment and technical details for each commit
- Export summaries to markdown files

### 🔄 Flexible AI Provider Support
- OpenAI (GPT-4) for high-quality results
- Ollama (local AI) for offline/private use
- Automatic fallback between providers
- Easy provider switching

## Installation

1. Ensure you have Python 3.7+ installed
2. Install the package using pip:
```bash
pip install sayless
```

## Configuration

### Using OpenAI (Recommended)
```bash
# Set your OpenAI API key
sayless config --openai-key YOUR_API_KEY

# Or use the quick switch command
sayless switch openai --key YOUR_API_KEY
```

### Using Ollama (Local AI)
1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Switch to Ollama:
```bash
sayless switch ollama
```

## Usage

### Generate Commit Messages
```bash
# Stage your changes first
git add .

# Generate and create a commit
sayless generate

# Preview without committing
sayless generate --preview
```

### Search Commits
```bash
# Search for commits related to a topic
sayless search "fix authentication bug"

# Index all commits before searching
sayless search "api improvements" --index-all

# Limit search results
sayless search "database optimization" --limit 10
```

### Analyze Changes
```bash
# Get detailed analysis of a specific commit
sayless summary <commit-hash> --detailed

# Summarize changes since last week
sayless since 1w

# Summarize changes between dates
sayless since 2023-01-01 --until 2023-12-31

# Save summary to file
sayless since 1m --save
```

### Configuration Commands
```bash
# Show current configuration
sayless config --show

# Switch AI providers
sayless switch openai --key YOUR_API_KEY
sayless switch ollama

# Change models
sayless switch openai --model gpt-4
sayless switch ollama --model llama2
```

## Requirements

- Python 3.7+
- Git
- For OpenAI: Valid API key and internet connection
- For Ollama: Local Ollama installation

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (alternative to using --openai-key)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - feel free to use this in your own projects!

## Support

If you encounter any issues or have questions, please open an issue on GitHub. 