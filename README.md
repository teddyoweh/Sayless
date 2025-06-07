# Sayless

Your intelligent Git companion that streamlines repository workflows through AI. Sayless handles everything from commit messages to PR management, making version control more intuitive and efficient.

## Core Capabilities

### Intelligent Git Operations
- Generate precise commit messages that capture the essence of your changes
- Create meaningful branch names based on your work context
- Get instant PR descriptions that highlight key changes and impact
- Search your entire commit history using natural language

### Repository Insights
- Get clear summaries of codebase changes over time
- Understand the impact of specific commits or PRs
- Track development progress with automated reports
- Export insights for team reviews and documentation

### Flexible AI Integration
- Use OpenAI for cloud-based processing
- Run everything locally with Ollama for complete privacy
- Switch between providers seamlessly based on your needs

## Quick Start

```bash
# Install
pip install sayless

# Configure with OpenAI
sayless config --openai-key YOUR_API_KEY

# Or use Ollama locally
sayless switch ollama
```

## Daily Workflow

```bash
# Generate commits
sl g                    # Generate commit from staged changes
sl g -a                 # Stage all changes and commit

# Branch management
sl branch "Add auth"    # Create and switch to a contextual branch
sl branches            # List branches with summaries

# Pull requests
sl pr create           # Create a PR with smart title and description
sl pr list --details   # List PRs with AI insights

# Search and analyze
sl search "auth fixes"  # Find relevant commits
sl since 1w            # Summarize last week's changes
```

## Requirements
- Python 3.7+
- Git
- OpenAI API key (for cloud features) or Ollama (for local processing)

## Contributing
We welcome contributions! Check out our [contribution guidelines](CONTRIBUTING.md) to get started.

## License
MIT License

## Support
Open an issue on GitHub for bugs or feature requests. For usage questions, check our [documentation](https://sayless.dev). 