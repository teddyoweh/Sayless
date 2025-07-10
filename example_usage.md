# Intelligent Dependency Evolution - Demo

The new dependency management feature provides intelligent analysis and automated updates for Node.js and Python dependencies.

## 🚀 Features

### 1. **Smart Dependency Analysis**
Analyze changes in commits to understand dependency impacts AND detect missing dependencies from code:

```bash
# Analyze current changes (detects both changes + missing deps)
sayless deps analyze

# Analyze specific commit
sayless deps analyze --commit abc123

# Auto-add missing dependencies to dependency files
sayless deps analyze --auto-fix

# Quick alias
sayless analyze-deps
```

**What it detects:**
- **Dependency file changes**: Updates to package.json, requirements.txt, etc.
- **Missing dependencies**: Import statements without corresponding package entries
- **Intelligent mapping**: `import requests` → `requests` package, `import cv2` → `opencv-python`
- **Version suggestions**: Latest stable versions from npm/PyPI registries

### 2. **Dependency Update Checking**
Check for available updates across ecosystems:

```bash
# Check all dependencies
sayless deps check

# Check specific ecosystem
sayless deps check --ecosystem npm
sayless deps check --ecosystem pip
sayless deps check --ecosystem poetry
```

### 3. **AI-Powered Auto Updates**
Intelligently update dependencies with breaking change detection:

```bash
# Preview what would be updated (dry run)
sayless deps update

# Actually perform updates
sayless deps update --auto-fix

# Update specific ecosystem
sayless deps update --ecosystem npm --auto-fix
```

## 🎯 What It Detects

### Node.js (npm)
- `package.json` changes
- `package-lock.json` updates
- `yarn.lock` modifications
- Breaking changes in major versions
- Security implications

### Python (pip/poetry)
- `requirements.txt` changes
- `pyproject.toml` updates
- Version constraint modifications
- Poetry dependency updates
- Virtual environment compatibility

## 🤖 AI Intelligence

The system provides:

### Change Analysis
- **Impact Assessment**: Security, performance, and compatibility implications
- **Breaking Change Detection**: Major version updates and API changes
- **Risk Classification**: High/medium/low risk categorization

### Update Planning
- **Intelligent Ordering**: Safe update sequence
- **Code Adaptation**: Specific suggestions for breaking changes
- **Testing Strategy**: What to test after updates
- **Rollback Plans**: How to revert if issues occur

### Framework-Specific Insights
- **React**: Prop type changes, lifecycle updates
- **Express**: Middleware signature changes
- **Django**: Model field changes, URL configuration
- **Flask**: Decorator changes, request context updates

## 📊 Example Output

```bash
$ sayless deps analyze

🔍 Dependency Changes Detected

📦 NPM Dependencies
Package              Change      Version
react               ⬆️ updated  17.0.2 → 18.2.0
express             ✅ added     +4.18.2
lodash              ❌ removed   -4.17.21

🔎 Missing Dependencies Detected

📦 PIP Missing Dependencies
Package        Import Statement           File             Confidence
requests       import requests            api/client.py    100%
pandas         import pandas as pd        data/process.py  100%
flask          from flask import Flask    app.py           100%

💡 Auto-add preview (3 packages):
  • Would add requests>=2.31.0
  • Would add pandas>=2.1.0
  • Would add flask>=3.0.0

To automatically add these dependencies:
   sayless deps analyze --auto-fix

🤖 AI Analysis & Recommendations

React 18 introduces significant breaking changes including:
- New concurrent features that may affect component behavior
- Stricter mode changes that could expose existing issues
- Updated lifecycle patterns requiring code review

Additionally, detected 3 missing dependencies from code analysis:
• requests (from api/client.py)
• pandas (from data/process.py)
• flask (from app.py)

Recommendations:
1. Test all React components thoroughly
2. Review for deprecated lifecycle methods
3. Update testing frameworks for React 18 compatibility
4. Run with --auto-fix to add missing Python dependencies
```

## 🛠️ Integration with Git Workflow

The dependency manager integrates seamlessly with your git workflow:

1. **Make Changes**: Update your `package.json`, `requirements.txt`, etc.
2. **Commit**: Use `sayless g -a` to create AI commit messages
3. **Analyze**: Run `sayless deps analyze` to understand impacts
4. **Update**: Use `sayless deps update --auto-fix` for automated updates
5. **Review**: Get AI-powered update plans and code adaptation suggestions

## 🔄 Supported Ecosystems

| Ecosystem | Files Detected | Update Support | Auto-Fix |
|-----------|---------------|----------------|----------|
| **npm** | `package.json`, `package-lock.json`, `yarn.lock` | ✅ | ✅ |
| **pip** | `requirements.txt`, `requirements-dev.txt` | ✅ | ⚠️ Manual |
| **poetry** | `pyproject.toml`, `poetry.lock` | ✅ | ✅ |

## 🎯 Use Cases

### 1. **Security Updates**
```bash
# Detect security vulnerabilities in dependencies
sayless deps analyze
# AI will highlight security implications and recommend immediate updates
```

### 2. **Major Version Migrations**
```bash
# Planning major framework updates
sayless deps analyze --commit migration-commit
# Get detailed migration plan with code adaptation steps
```

### 3. **Automated Maintenance**
```bash
# Regular dependency maintenance
sayless deps check
sayless deps update --auto-fix
# Keep dependencies current with intelligent conflict resolution
```

### 4. **Code Review Support**
```bash
# During PR reviews
sayless deps analyze --commit pr-commit
# Understand dependency changes impact for better reviews
```

### 5. **Intelligent Dependency Resolution**
```bash
# After adding new imports to your code
git add . 
sayless deps analyze --auto-fix
# Automatically detects missing packages and adds them to dependency files

# Smart import-to-package mapping
# import cv2 → opencv-python
# import PIL → Pillow  
# import sklearn → scikit-learn
# import jwt → PyJWT
```

This feature transforms dependency management from a manual, error-prone process into an intelligent, automated workflow that anticipates issues and provides actionable guidance. 