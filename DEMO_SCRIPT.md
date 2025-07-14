# Sayless Complete Demo & Usage Guide

Welcome to the complete Sayless demo! This guide walks you through all features with practical examples and real-world workflows.

## 🚀 Getting Started

### Installation & Setup
```bash
# Install Sayless
pip install sayless

# Quick setup with OpenAI (recommended)
sayless config --openai-key YOUR_API_KEY

# Alternative: Local AI with Ollama
# 1. Install Ollama from https://ollama.ai
# 2. Switch to local processing
sayless switch ollama
```

### Verify Your Setup
```bash
# Check configuration
sl config --show

# Test with a simple command
sl --help
```

## 🎯 Core Workflows

### 1. Smart Commit Generation

#### Basic Usage
```bash
# Scenario: You've added a new login function
echo "def login(username, password): return authenticate(username, password)" > auth.py
git add auth.py

# Generate intelligent commit
sl g
# ✨ Result: "feat(auth): add user login function with authentication"

# Or stage everything and commit in one command
sl g -a
```

#### Advanced Commit Examples
```bash
# Different types of changes Sayless recognizes:

# 1. Bug Fix
echo "# Fix: Handle null pointer exception" >> bug_fix.py
sl g -a
# → "fix(validation): handle null pointer exception in user input"

# 2. Feature Addition
mkdir -p api/endpoints
echo "def create_user(): pass" > api/endpoints/users.py
sl g -a
# → "feat(api): add user creation endpoint"

# 3. Documentation Update
echo "# Installation Guide" > docs/install.md
sl g -a
# → "docs: add installation guide"

# 4. Refactoring
echo "# Simplified authentication logic" >> auth.py
sl g -a
# → "refactor(auth): simplify authentication logic"

# Preview before committing
sl g --preview
# Shows what the commit message would be without committing
```

### 2. Intelligent Branch Management

#### Create Branches from Descriptions
```bash
# Sayless creates meaningful branch names from your descriptions

sl branch "Add user authentication with JWT tokens"
# → Creates and switches to: feat/add-user-authentication-jwt-tokens

sl branch "Fix memory leak in image processing"
# → Creates: fix/memory-leak-image-processing

sl branch "Update documentation for API endpoints"
# → Creates: docs/update-documentation-api-endpoints
```

#### Generate Branch Names from Changes
```bash
# Make changes first, then let AI suggest the branch name
echo "class PaymentProcessor:" > payment.py
echo "    def process_payment(self, amount): pass" >> payment.py
git add payment.py

# Generate branch name from staged changes
sl branch -g
# AI analyzes changes and suggests: feat/add-payment-processor

# Combine staging and branch generation
sl branch -g -a
# Stages all changes AND creates branch based on them
```

#### Branch Analysis
```bash
# List branches with AI-generated summaries
sl branches --details

# Example output:
# main               Main development branch
# feat/user-auth     Implements JWT authentication system
# fix/login-bug      Resolves login validation issues
# docs/api-guide     Updates API documentation
```

### 3. AI-Powered Pull Requests

#### Basic PR Creation
```bash
# Create PR with AI-generated title, description, and labels
sl pr create

# Example generated PR:
# Title: "feat(auth): implement JWT authentication system"
# 
# ## Overview
# This PR implements a comprehensive JWT authentication system for user login and session management.
#
# ## Changes
# - Add JWT token generation and validation
# - Implement user login endpoint
# - Add middleware for protected routes
# - Include comprehensive test coverage
#
# ## Testing
# - Run `pytest tests/test_auth.py` for authentication tests
# - Test login flow with sample credentials
# - Verify token expiration handling
#
# Labels: [authentication, security, feature]
```

#### Advanced PR Options
```bash
# Target specific branch
sl pr create --base develop

# Include detailed analysis
sl pr create --details

# Create PR for different base
sl pr create --base release/v2.0
```

#### PR Management
```bash
# List open PRs with AI insights
sl pr list --details

# Example output:
# #123  feat(auth): JWT implementation    john_dev    auth-feature
#       → Adds comprehensive authentication system with security focus
#
# #124  fix(api): resolve timeout issues  jane_dev    fix-timeouts  
#       → Critical bug fix for API response timeouts
```

### 4. Code Review Assistant

#### Review Your Current Work
```bash
# Review current branch changes
sl review

# Example output:
# 🤖 Code Review - feat/user-auth
# 
# The authentication implementation looks solid overall. You've added proper JWT 
# handling with token validation and expiration checks. The login function correctly 
# handles password hashing, and I like how you've separated the authentication logic 
# into its own module. One thing to consider is adding rate limiting to prevent 
# brute force attacks on the login endpoint. Also, make sure to validate the JWT 
# secret is stored securely and not hardcoded.
```

#### Review Specific PRs
```bash
# Review a specific PR
sl review --pr 123

# Post review directly to GitHub
sl review --pr 123 --auto-comment

# Review with automatic posting for current branch
sl review --auto-comment
```

#### Enhanced Review Types
```bash
# Security-focused review
sl review-enhanced --pr 123 --type security

# Performance-focused review  
sl review-enhanced --pr 123 --type performance

# Comprehensive detailed review
sl review-enhanced --pr 123 --type detailed

# Dependency-focused review
sl review-enhanced --pr 123 --type dependencies

# Quick overview review
sl review-enhanced --pr 123 --type quick
```

#### Bulk Reviews
```bash
# Review multiple PRs at once
sl bulk-review --type quick --max 5

# Auto-post reviews to GitHub
sl bulk-review --type security --max 3 --auto-post
```

### 5. Semantic Repository Search

#### Natural Language Search
```bash
# Search using natural language instead of git log grep
sl search "authentication improvements"
sl search "API endpoint changes"
sl search "React component updates"
sl search "database migration scripts"
sl search "performance optimizations in image processing"

# Limit results
sl search "bug fixes" --limit 10

# Example output:
# 🔍 Found 5 relevant commits for "authentication improvements":
#
# abc123f  feat(auth): implement JWT authentication system
#          2 days ago by john_dev
#          Added comprehensive JWT auth with token validation
#
# def456a  fix(auth): resolve login validation bug  
#          1 week ago by jane_dev
#          Fixed edge case in password validation logic
```

#### Advanced Search Options
```bash
# Index all commits for better search (one-time setup)
sl search "database changes" --index-all

# Search with context
sl search "React hooks implementation" --context
```

### 6. Repository Analysis & Insights

#### Time-based Analysis
```bash
# Analyze recent changes
sl since 1d              # Last 24 hours
sl since 1w              # Last week
sl since 1m              # Last month
sl since 3m              # Last 3 months

# Analyze from specific date
sl since 2023-06-01      # All changes since June 1st
sl since 2023-01-01      # Year-to-date changes

# Example output:
# 📊 Repository Analysis - Last Week
#
# ## Summary
# 15 commits across 8 files with focus on authentication and API improvements
#
# ## Key Changes
# - Implemented JWT authentication system (5 commits)
# - Fixed critical API timeout issues (3 commits)  
# - Updated documentation (2 commits)
# - Performance optimizations (5 commits)
#
# ## Impact
# +324 lines, -156 lines across authentication, API, and documentation modules
```

#### Export Analysis
```bash
# Save analysis to markdown file
sl since 1w --save
# Creates: sayless_analysis_YYYY-MM-DD.md

# Save with custom name
sl since 1m --save --filename monthly_report.md
```

#### Specific Commit Analysis
```bash
# Analyze a specific commit
sl summary abc123f --detailed

# Example output:
# 🔍 Commit Analysis: abc123f
#
# Title: feat(auth): implement JWT authentication system
# Author: john_dev
# Date: 2 days ago
#
# ## Impact Analysis
# This commit introduces a comprehensive authentication system using JWT tokens.
# Key additions include token generation, validation middleware, and secure
# session management. The implementation follows security best practices with
# proper token expiration and refresh mechanisms.
#
# ## Files Changed
# - auth/jwt_handler.py (new): JWT token operations
# - middleware/auth.py (modified): Authentication middleware
# - tests/test_auth.py (new): Comprehensive test coverage
```

### 7. Dependency Management

#### Analyze Dependencies
```bash
# Analyze dependency changes in current commit
sl deps analyze

# Example output:
# 🔍 Dependency Changes Detected
#
# 📦 NPM Dependencies
# Package              Change      Version
# express             ⬆️ updated  4.17.1 → 4.18.2
# jsonwebtoken        ✅ added     +9.0.2
# bcrypt              ✅ added     +5.1.0
#
# 🔎 Missing Dependencies Detected  
# Package        Import Statement           File             Confidence
# requests       import requests            api/client.py    100%
# pandas         import pandas as pd        data/process.py  100%
#
# 🤖 AI Analysis
# The Express.js update includes security patches and performance improvements.
# The addition of jsonwebtoken and bcrypt suggests implementation of secure
# authentication. Consider updating related packages like passport for consistency.
#
# To auto-add missing dependencies: sl deps analyze --auto-fix
```

#### Auto-fix Missing Dependencies
```bash
# Automatically add missing dependencies
sl deps analyze --auto-fix

# Check for available updates
sl deps check

# Update dependencies with AI guidance
sl deps update --auto-fix
```

## 🔄 Complete Development Workflows

### Workflow 1: Feature Development
```bash
# 1. Create feature branch
sl branch "Add user profile management"
# → feat/add-user-profile-management

# 2. Make your changes
echo "class UserProfile:" > profile.py
echo "    def update_profile(self): pass" >> profile.py

# 3. Stage and commit with AI
sl g -a
# → "feat(profile): add user profile management class"

# 4. Continue development
echo "    def get_profile(self): pass" >> profile.py
sl g -a
# → "feat(profile): add profile retrieval method"

# 5. Create PR when ready
sl pr create
# → AI generates comprehensive PR with title, description, labels

# 6. Review before merging
sl review
# → Get AI feedback on your changes
```

### Workflow 2: Bug Fix Process
```bash
# 1. Create bug fix branch
sl branch "Fix login validation error"
# → fix/login-validation-error

# 2. Implement fix
echo "# Fixed null validation bug" >> auth.py
sl g -a
# → "fix(auth): handle null validation in login process"

# 3. Create PR with detailed analysis
sl pr create --details

# 4. Review the fix
sl review-enhanced --type security
# → Security-focused review of the bug fix
```

### Workflow 3: Code Review Process
```bash
# 1. Review incoming PRs
sl pr list --details
# See AI insights for all open PRs

# 2. Detailed review of specific PR
sl review-enhanced --pr 123 --type detailed

# 3. Security review for sensitive changes
sl review-enhanced --pr 124 --type security --auto-post

# 4. Bulk review multiple PRs
sl bulk-review --type quick --max 5
```

### Workflow 4: Repository Maintenance
```bash
# 1. Weekly repository analysis
sl since 1w --save
# Get comprehensive weekly report

# 2. Dependency maintenance
sl deps check
sl deps update --auto-fix

# 3. Search for specific patterns
sl search "TODO comments"
sl search "deprecated function usage"

# 4. Analyze impact of major changes
sl summary abc123f --detailed
```

## 🛠️ Advanced Configuration

### AI Provider Management
```bash
# Switch between providers
sl switch openai --model gpt-4
sl switch openai --model gpt-3.5-turbo
sl switch ollama --model llama2
sl switch ollama --model codellama

# Configure provider-specific settings
sl config --openai-key YOUR_KEY
sl config --github-token YOUR_TOKEN
```

### Customization Options
```bash
# View all configuration
sl config --show

# Set preferences
export OPENAI_API_KEY=your_key
export GITHUB_TOKEN=your_token
export SAYLESS_DEFAULT_MODEL=gpt-4
```

## 🎯 Pro Tips

### 1. Commit Message Quality
- Stage related changes together for focused commits
- Use `sl g --preview` to review messages before committing
- Let Sayless detect the type (feat, fix, docs, etc.) automatically

### 2. Branch Organization
- Use descriptive branch names that Sayless can create
- Let AI generate names from your changes with `sl branch -g`
- Use `sl branches --details` to track branch purposes

### 3. PR Management
- Create PRs early with `sl pr create` for better collaboration
- Use `--details` flag for complex changes
- Review your own work with `sl review` before requesting reviews

### 4. Code Reviews
- Use different review types for different concerns
- Leverage `--auto-comment` for consistent review posting
- Combine quick reviews with detailed analysis as needed

### 5. Repository Insights
- Run weekly analysis with `sl since 1w --save`
- Use semantic search to find specific patterns
- Export analysis for team meetings and documentation

## 🔍 Troubleshooting

### Common Issues

**1. AI Provider Issues**
```bash
# Check configuration
sl config --show

# Test connectivity
sl g --preview  # Should work if API is configured

# Switch providers if needed
sl switch ollama  # Fallback to local AI
```

**2. Git Integration Issues**
```bash
# Verify git status
git status

# Ensure you're in a git repository
git init  # If needed

# Check remote configuration
git remote -v
```

**3. GitHub Integration Issues**
```bash
# Verify token permissions
sl pr list  # Should work if token is valid

# Set token if needed
sl config --github-token YOUR_TOKEN

# Check repository access
git remote get-url origin
```

**4. Performance Optimization**
```bash
# For large repositories, use limits
sl search "query" --limit 10
sl since 1w  # Instead of analyzing entire history

# Use local AI for privacy/speed
sl switch ollama
```

This comprehensive guide covers all major Sayless features. Start with the basic workflows and gradually incorporate advanced features as you become more comfortable with the tool!

---

*Made with ❤️ for developers who want smarter Git workflows* 