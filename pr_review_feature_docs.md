# PR Review Feature Documentation

## Overview

The enhanced PR review feature provides AI-powered, structured code reviews with multiple review types, automated checklists, and comprehensive analysis. It integrates with the existing dependency management system to provide holistic code review capabilities.

## Key Features

### 🔍 Multiple Review Types
- **Quick Review**: Fast overview for basic code quality and functionality
- **Detailed Review**: Comprehensive technical analysis covering architecture, security, and performance
- **Security Review**: Focused security vulnerability assessment
- **Performance Review**: Performance and scalability analysis
- **Dependencies Review**: Integrated dependency change analysis

### 📋 Automated Checklists
- Dynamic checklist generation based on review type
- Intelligent status assessment based on code changes
- Visual status indicators (✅ ⚠️ ❌ 👀 ❓)

### 🤖 AI-Powered Analysis
- Structured AI prompts for consistent reviews
- Confidence scoring for review reliability
- Fallback mechanisms for robust operation

### 🔄 Bulk Operations
- Review multiple PRs simultaneously
- Compare different review types for the same PR
- Batch processing with progress tracking

## Commands

### Enhanced Review Commands

#### `sayless review-enhanced`
Enhanced PR review with structured review types.

```bash
# Review a specific PR with quick analysis
sayless review-enhanced --pr 123

# Detailed technical review
sayless review-enhanced --pr 123 --type detailed

# Security-focused review
sayless review-enhanced --pr 123 --type security --auto-post

# Review current branch
sayless review-enhanced --current --type performance

# Review without checklist
sayless review-enhanced --pr 123 --no-checklist
```

**Options:**
- `--pr <number>`: PR number to review
- `--type <type>`: Review type (quick, detailed, security, performance, dependencies)
- `--auto-post`: Automatically post review to GitHub
- `--checklist/--no-checklist`: Include/exclude review checklist
- `--current`: Review current branch

#### `sayless bulk-review`
Review multiple open PRs at once.

```bash
# Quick review of up to 5 PRs
sayless bulk-review

# Detailed review of up to 10 PRs
sayless bulk-review --type detailed --max 10

# Auto-post security reviews
sayless bulk-review --type security --auto-post
```

**Options:**
- `--type <type>`: Review type for all PRs
- `--max <number>`: Maximum number of PRs to review (default: 5)
- `--auto-post`: Automatically post all reviews to GitHub

#### `sayless compare-reviews`
Compare multiple review types for the same PR.

```bash
# Compare all review types for PR #123
sayless compare-reviews 123
```

This command runs quick, detailed, security, and performance reviews on the same PR and displays a comparison table.

#### `sayless review-templates`
Show available review templates and their details.

```bash
sayless review-templates
```

Displays all available review types, their focus areas, and sample checklist items.

#### `sayless review` (Enhanced)
Original review command with enhanced capabilities.

```bash
# Original quick review
sayless review --pr 123

# Enhanced structured review
sayless review --pr 123 --enhanced --type detailed

# Review current branch with enhancement
sayless review --enhanced --type security
```

**Options:**
- `--enhanced`: Use enhanced structured review system
- `--type <type>`: Review type (when using enhanced mode)

## Review Types

### 1. Quick Review (`quick`)
**Purpose**: Fast overview for basic validation
**Focus Areas**: functionality, code_style, tests
**Checklist**:
- Code follows project conventions
- Changes are focused and minimal
- No obvious bugs or issues
- Tests are included if needed

### 2. Detailed Review (`detailed`)
**Purpose**: Comprehensive technical analysis
**Focus Areas**: architecture, maintainability, performance, security, testing
**Checklist**:
- Architecture and design patterns
- Code maintainability and readability
- Error handling and edge cases
- Performance implications
- Security considerations
- Test coverage and quality
- Documentation updates
- Breaking changes identified

### 3. Security Review (`security`)
**Purpose**: Security vulnerability assessment
**Focus Areas**: security, validation, encryption, access_control
**Checklist**:
- Input validation and sanitization
- Authentication and authorization
- Data encryption and storage
- SQL injection prevention
- XSS protection
- CSRF protection
- Secrets and sensitive data handling
- Access control implementation

### 4. Performance Review (`performance`)
**Purpose**: Performance and scalability analysis
**Focus Areas**: performance, scalability, optimization, monitoring
**Checklist**:
- Algorithm efficiency
- Database query optimization
- Memory usage patterns
- Caching strategies
- API response times
- Resource utilization
- Scalability considerations
- Performance monitoring

### 5. Dependencies Review (`dependencies`)
**Purpose**: Dependency change analysis with security assessment
**Focus Areas**: dependencies, security, compatibility, licensing
**Checklist**:
- New dependencies justified
- Version compatibility checked
- Security vulnerabilities in deps
- Bundle size impact
- License compatibility
- Maintenance status of deps
- Alternative solutions considered
- Dependency conflicts resolved

**Special Features**:
- Integrates with dependency manager
- Analyzes dependency changes from commits
- Detects missing dependencies in code
- Checks for breaking changes
- Security vulnerability assessment

## Integration with Dependency Management

The dependencies review type integrates seamlessly with the existing dependency management system:

1. **Automatic Detection**: Detects dependency changes from git commits
2. **Missing Dependencies**: Finds packages used in code but not in dependency files
3. **Security Analysis**: Identifies security-related dependency changes
4. **Breaking Changes**: Warns about potential breaking changes in version updates
5. **AI Insights**: Combines dependency analysis with AI-generated recommendations

## Output Formats

### Review Result Display
Each review includes:
- **Review Type**: The type of review conducted
- **Assessment**: APPROVE ✅ / REQUEST CHANGES ❌ / COMMENT 💬
- **Confidence Score**: AI confidence in the review (0-100%)
- **Summary**: Brief overview of the changes
- **Detailed Feedback**: Comprehensive analysis and recommendations

### Checklist Display
- **Status Icons**: Visual indicators for each checklist item
- **Notes**: Contextual information about each item
- **Automated Assessment**: Intelligent status detection based on code changes

### Bulk Review Summary
- **Table Format**: Easy-to-scan results table
- **Assessment Overview**: Quick view of all PR assessments
- **Confidence Tracking**: Confidence scores for all reviews

## GitHub Integration

### Posting Reviews
Reviews can be automatically posted to GitHub with:
- Structured formatting with markdown
- Emoji indicators and sections
- Confidence scoring
- Checklist integration

### Review Events
The system posts appropriate GitHub review events:
- `APPROVE`: For positive assessments
- `REQUEST_CHANGES`: For issues requiring fixes
- `COMMENT`: For general feedback

## Usage Examples

### Basic Workflow
```bash
# 1. Quick review of a PR
sayless review-enhanced --pr 123

# 2. If issues found, do detailed review
sayless review-enhanced --pr 123 --type detailed

# 3. Post security review if sensitive changes
sayless review-enhanced --pr 123 --type security --auto-post
```

### Team Review Process
```bash
# 1. Bulk review all open PRs
sayless bulk-review --max 10

# 2. Detailed review of complex PRs
sayless review-enhanced --pr 456 --type detailed --auto-post

# 3. Compare review types for critical PR
sayless compare-reviews 789
```

### Dependency-Focused Review
```bash
# Review PR with focus on dependencies
sayless review-enhanced --pr 123 --type dependencies --auto-post

# Check current branch for dependency issues
sayless review-enhanced --current --type dependencies
```

## Configuration

The PR review feature uses the same configuration as other sayless features:
- **AI Provider**: OpenAI or Ollama
- **GitHub Token**: Required for posting reviews
- **Model Settings**: Configurable AI model

## Best Practices

### When to Use Each Review Type
- **Quick**: Small PRs, hotfixes, documentation changes
- **Detailed**: Large features, architectural changes, complex logic
- **Security**: Authentication changes, data handling, external integrations
- **Performance**: Database changes, algorithms, high-traffic features
- **Dependencies**: Package updates, new dependencies, security patches

### Review Workflow Integration
1. Use bulk review for daily PR triage
2. Apply specific review types based on PR content
3. Compare review types for critical changes
4. Post reviews to GitHub for team visibility

### Confidence Score Interpretation
- **90-100%**: High confidence, can auto-post
- **70-89%**: Good confidence, review before posting
- **50-69%**: Medium confidence, manual review recommended
- **Below 50%**: Low confidence, manual review required

## Error Handling

The system includes robust error handling:
- **Fallback AI Providers**: Ollama fallback for OpenAI failures
- **Large Diff Handling**: Intelligent truncation for large changes
- **Network Issues**: Graceful degradation with error messages
- **Invalid PRs**: Clear error messages for missing/invalid PRs

## Future Enhancements

Planned improvements include:
- Custom review templates
- Team-specific checklists
- Integration with CI/CD systems
- Historical review analytics
- Machine learning improvements based on feedback 