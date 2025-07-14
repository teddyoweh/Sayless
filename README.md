# Sayless

> Your AI-powered Git companion that makes version control intelligent and effortless

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

Sayless transforms your Git workflow with AI that understands your code. Generate meaningful commits, create smart pull requests, conduct code reviews, and search your repository using natural language.

## ✨ What Makes Sayless Special

- **🤖 Intelligent Commits**: Generate precise commit messages that actually explain your changes
- **🔀 Smart PRs**: Create pull requests with contextual titles, descriptions, and labels
- **🔍 AI Code Review**: Get detailed, actionable feedback on your code changes
- **🔎 Semantic Search**: Find commits using natural language instead of grep
- **📊 Repository Insights**: Understand your project's evolution over time
- **🔒 Privacy Options**: Use OpenAI or run everything locally with Ollama

## 🚀 Quick Start

### Installation
```bash
pip install sayless
```

### Setup (Choose one)
**Option 1: OpenAI (Recommended)**
```bash
sayless config --openai-key YOUR_API_KEY
```

**Option 2: Local AI with Ollama**
```bash
# Install Ollama from https://ollama.ai first
sayless switch ollama
```

### Your First AI Commit
```bash
# Make some changes to your code
echo "print('Hello, AI!')" > hello.py

# Generate an intelligent commit
sayless g -a
# ✨ Result: "feat: add hello world example script"
```

That's it! Sayless analyzed your changes and created a proper commit message.

## 🎯 Core Features

### 1. Intelligent Commit Generation
Never write another commit message manually:

```bash
# Stage changes and generate commit
git add .
sl g                    # Generate commit message

# One command to stage all and commit
sl g -a                 # Stage all changes + commit

# Preview before committing
sl g --preview          # Show message without committing
```

**Example**: Add a login function → `"feat(auth): implement user login with JWT validation"`

### 2. Smart Branch Management
Create meaningful branch names instantly:

```bash
# Create branch from description
sl branch "Fix memory leak in image processor"
# → Creates: fix/memory-leak-image-processor

# Generate branch name from your changes
git add .
sl branch -g            # AI analyzes changes and suggests name

# List branches with AI summaries
sl branches --details
```

### 3. AI-Powered Pull Requests
Create PRs that reviewers will love:

```bash
# Create PR with AI-generated content
sl pr create

# Example generated PR:
# Title: "feat(api): add user authentication with JWT"
# Body: Detailed description with testing instructions
# Labels: [authentication, security, feature]
```

### 4. Code Review Assistant
Get instant, detailed code reviews:

```bash
# Review current changes
sl review

# Review specific PR
sl review --pr 123

# Post review to GitHub
sl review --pr 123 --auto-comment

# Advanced review types
sl review-enhanced --pr 123 --type security     # Security-focused
sl review-enhanced --pr 123 --type performance  # Performance-focused
```

### 5. Semantic Repository Search
Find what you need using natural language:

```bash
# Search your commit history
sl search "authentication fixes"
sl search "performance improvements in database"
sl search "bug fixes related to user login"

# Get repository summaries
sl since 1w             # Summarize last week
sl since 1m             # Summarize last month
```

### 6. Repository Analysis
Understand your project's evolution:

```bash
# Analyze specific timeframes
sl since 2023-01-01     # Changes since date
sl since 1w --save      # Save analysis to file

# Dependency insights
sl deps analyze         # Analyze dependency changes
sl deps check           # Check for updates
```

## 📖 Detailed Usage

<details>
<summary><strong>Commit Message Best Practices</strong></summary>

Sayless follows conventional commit formats and adapts to your project:

```bash
# Different types of changes
sl g -a  # Adding feature    → "feat(module): add new feature"
sl g -a  # Fixing bug        → "fix(component): resolve issue with X"
sl g -a  # Updating docs     → "docs(readme): update installation guide"
sl g -a  # Refactoring       → "refactor(auth): simplify login logic"
```

**Tips:**
- Stage related changes together for focused commits
- Use `--preview` to review messages before committing
- Let Sayless handle the formatting and scope detection

</details>

<details>
<summary><strong>Advanced Branch Workflows</strong></summary>

```bash
# Branch creation options
sl branch "Add dark mode" --no-checkout    # Create without switching
sl branch -g -a                           # Stage all + generate name
sl branch "Update deps" --base develop     # Custom base branch

# Branch analysis
sl branches                                # List with basic info
sl branches --details                      # List with AI insights
```

</details>

<details>
<summary><strong>Pull Request Management</strong></summary>

```bash
# PR creation
sl pr create                               # Basic PR creation
sl pr create --base develop                # Target specific branch
sl pr create --details                     # Include detailed analysis

# PR review and management  
sl pr list                                 # List open PRs
sl pr list --details                       # List with AI insights

# Advanced PR reviews
sl review-enhanced --pr 123 --type detailed    # Comprehensive review
sl review-enhanced --pr 123 --type security    # Security-focused review
sl bulk-review --type quick --max 5            # Review multiple PRs
```

</details>

<details>
<summary><strong>Search and Analysis</strong></summary>

```bash
# Semantic search examples
sl search "API endpoint changes"
sl search "React component updates"
sl search "database migration scripts"
sl search "security vulnerability fixes" --limit 5

# Repository analysis
sl since 1d                                # Last 24 hours
sl since 1w                                # Last week  
sl since 1m                                # Last month
sl since 2023-06-01                        # Since specific date

# Export and save
sl since 1w --save                         # Export to markdown
```

</details>

## ⚙️ Configuration

### AI Provider Setup

**OpenAI (Cloud)**
```bash
# Configure OpenAI
sl config --openai-key YOUR_API_KEY
sl switch openai --model gpt-4             # Use GPT-4

# Environment variable option
export OPENAI_API_KEY=your_api_key
```

**Ollama (Local)**
```bash
# Install Ollama first: https://ollama.ai
sl switch ollama                           # Switch to local AI
sl switch ollama --model llama2            # Use specific model
```

### GitHub Integration
```bash
# Configure GitHub token for PR operations
sl config --github-token YOUR_TOKEN

# Or use environment variable
export GITHUB_TOKEN=your_github_token
```

### View Current Configuration
```bash
sl config --show                           # Display current settings
```

## 🔧 Command Reference

| Command | Description | Example |
|---------|-------------|---------|
| `sl g` | Generate commit message | `sl g -a` |
| `sl branch` | Create smart branch | `sl branch "Fix login bug"` |
| `sl pr create` | Create pull request | `sl pr create --details` |
| `sl review` | Review code changes | `sl review --pr 123` |
| `sl search` | Semantic search | `sl search "auth fixes"` |
| `sl since` | Repository analysis | `sl since 1w --save` |
| `sl deps` | Dependency management | `sl deps analyze` |
| `sl config` | Configuration | `sl config --show` |

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create your feature branch**: `sl branch "Add awesome feature"`
3. **Commit your changes**: `sl g -a`
4. **Push to your fork**: `git push origin feature-branch`
5. **Create a Pull Request**: `sl pr create`

For detailed contributing guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support & Community

- **Issues**: [GitHub Issues](https://github.com/yourusername/sayless/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/sayless/discussions)
- **Documentation**: [sayless.dev](https://sayless.dev) *(coming soon)*

## 🙏 Acknowledgments

- Thanks to all contributors who make Sayless better
- Inspired by the need for more intelligent development tools
- Built with love for the developer community

---

*Made with ❤️ by developers, for developers* 