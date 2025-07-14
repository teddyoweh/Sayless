import subprocess
import sys
from typing import List, Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import requests
import json
import os
import typer
from .ai_providers import OpenAIProvider, OllamaProvider
from .config import Config
from .git_ops import run_git_command, get_current_branch, get_default_branch
from .dependency_manager import DependencyManager
from dataclasses import dataclass
from enum import Enum

console = Console()
settings = Config()

class ReviewType(Enum):
    """Types of reviews available"""
    QUICK = "quick"          # Fast overview review
    DETAILED = "detailed"    # Comprehensive technical review
    SECURITY = "security"    # Security-focused review
    PERFORMANCE = "performance"  # Performance-focused review
    DEPENDENCIES = "dependencies"  # Dependency-focused review
    ACCESSIBILITY = "accessibility"  # Accessibility review
    CHECKLIST = "checklist"  # Checklist-based review

@dataclass
class ReviewResult:
    """Result of a PR review"""
    pr_number: int
    review_type: ReviewType
    assessment: str  # APPROVE, REQUEST_CHANGES, COMMENT
    summary: str
    detailed_feedback: str
    checklist_items: List[Dict] = None
    security_issues: List[str] = None
    performance_issues: List[str] = None
    dependency_issues: List[str] = None
    confidence_score: float = 0.0

@dataclass
class ReviewTemplate:
    """Template for structured reviews"""
    name: str
    review_type: ReviewType
    checklist: List[str]
    focus_areas: List[str]
    ai_prompt_template: str



class GitHubAPI:
    def __init__(self):
        self.token = settings.get_github_token()
        if not self.token:
            console.print(Panel(
                "[red]GitHub token not found[/red]\n\n"
                "[yellow]Please set your GitHub token using one of these methods:[/yellow]\n"
                "1. Configure via command:\n"
                "   [blue]sl config --github-token YOUR_TOKEN[/blue]\n"
                "2. Set environment variable:\n"
                "   [blue]export GITHUB_TOKEN=your_token[/blue]",
                title="Configuration Required",
                border_style="red"
            ))
            sys.exit(1)
        
        self.api_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Get repository info
        try:
            self.repo_url = run_git_command(['config', '--get', 'remote.origin.url']).stdout.strip()
            if "github.com" not in self.repo_url:
                raise ValueError("Not a GitHub repository")
            
            # Extract owner and repo from URL
            self.repo_url = self.repo_url.rstrip('.git')
            parts = self.repo_url.split('github.com/')[-1].split('/')
            self.owner = parts[-2]
            self.repo = parts[-1]
        except Exception as e:
            console.print(Panel(
                "[red]Failed to get repository information[/red]\n\n"
                "[yellow]Please ensure:[/yellow]\n"
                "1. You're in a git repository\n"
                "2. The repository has a GitHub remote\n"
                "3. You have access to the repository",
                title="Repository Error",
                border_style="red"
            ))
            sys.exit(1)

    def get_pr_by_number(self, pr_number: int) -> Dict:
        """Get PR details by number"""
        url = f"{self.api_url}/repos/{self.owner}/{self.repo}/pulls/{pr_number}"
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"PR #{pr_number} not found or inaccessible")
        return response.json()

    def get_pr_diff(self, pr_number: int) -> str:
        """Get the diff for a specific PR"""
        url = f"{self.api_url}/repos/{self.owner}/{self.repo}/pulls/{pr_number}"
        headers = {**self.headers, "Accept": "application/vnd.github.v3.diff"}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to get diff for PR #{pr_number}")
        return response.text

    def get_pr_files(self, pr_number: int) -> List[Dict]:
        """Get list of files changed in a PR"""
        url = f"{self.api_url}/repos/{self.owner}/{self.repo}/pulls/{pr_number}/files"
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"Failed to get files for PR #{pr_number}")
        return response.json()

    def post_pr_review(self, pr_number: int, body: str, event: str = "COMMENT", comments: List[Dict] = None) -> Dict:
        """Post a review on a PR"""
        url = f"{self.api_url}/repos/{self.owner}/{self.repo}/pulls/{pr_number}/reviews"
        data = {
            "body": body,
            "event": event  # COMMENT, APPROVE, REQUEST_CHANGES
        }
        if comments:
            data["comments"] = comments
        
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Failed to post review: {response.json().get('message', 'Unknown error')}")
        return response.json()

    def post_pr_comment(self, pr_number: int, body: str) -> Dict:
        """Post a general comment on a PR"""
        url = f"{self.api_url}/repos/{self.owner}/{self.repo}/issues/{pr_number}/comments"
        data = {"body": body}
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code != 201:
            raise Exception(f"Failed to post comment: {response.json().get('message', 'Unknown error')}")
        return response.json()

    def get_current_pr(self) -> Optional[Dict]:
        """Get PR for current branch if it exists"""
        current_branch = get_current_branch()
        url = f"{self.api_url}/repos/{self.owner}/{self.repo}/pulls"
        params = {"head": f"{self.owner}:{current_branch}", "state": "open"}
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code == 200 and response.json():
            return response.json()[0]
        return None

    def validate_pr_params(self, head: str, base: str) -> None:
        """Validate PR parameters before creation"""
        try:
            # Check if base branch exists
            url = f"{self.api_url}/repos/{self.owner}/{self.repo}/branches/{base}"
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                raise ValueError(f"Base branch '{base}' not found")
            
            # Check if head branch exists and has commits
            url = f"{self.api_url}/repos/{self.owner}/{self.repo}/branches/{head}"
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                raise ValueError(f"Current branch '{head}' not found on GitHub. Push your changes first:\n[blue]git push -u origin {head}[/blue]")
            
            # Check if PR already exists
            url = f"{self.api_url}/repos/{self.owner}/{self.repo}/pulls"
            params = {"head": f"{self.owner}:{head}", "base": base, "state": "open"}
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200 and response.json():
                existing_pr = response.json()[0]
                raise ValueError(
                    f"A PR already exists for this branch:\n"
                    f"#{existing_pr['number']} - {existing_pr['title']}\n"
                    f"URL: {existing_pr['html_url']}"
                )
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to validate PR parameters: {str(e)}")

    def create_pr(self, title: str, body: str, base: str = "main", labels: List[str] = None) -> Dict:
        """Create a pull request"""
        head = get_current_branch()
        
        try:
            # Try to validate parameters, but continue if it fails due to auth issues
            try:
                self.validate_pr_params(head, base)
            except ValueError as e:
                if "Bad credentials" in str(e) or "401" in str(e):
                    console.print("[yellow]⚠️  Skipping validation due to GitHub authentication issue[/yellow]")
                    console.print("[yellow]   The PR will still be created if your branch exists on GitHub[/yellow]")
                else:
                    raise e
            
            # Create PR
            url = f"{self.api_url}/repos/{self.owner}/{self.repo}/pulls"
            data = {
                "title": title,
                "body": body,
                "head": head,
                "base": base
            }
            
            response = requests.post(url, headers=self.headers, json=data)
            if response.status_code != 201:
                error_data = response.json()
                error_msg = error_data.get('message', '')
                errors = error_data.get('errors', [])
                error_details = '\n'.join(f"- {e.get('message', '')}" for e in errors) if errors else ''
                
                raise ValueError(
                    f"{error_msg}\n{error_details}\n\n"
                    "[yellow]Common issues:[/yellow]\n"
                    "1. Branch not pushed to GitHub\n"
                    "2. No commits in branch\n"
                    "3. Branch already has an open PR"
                )
            
            pr = response.json()
            
            # Add labels if provided
            if labels:
                labels_url = f"{self.api_url}/repos/{self.owner}/{self.repo}/issues/{pr['number']}/labels"
                requests.post(labels_url, headers=self.headers, json=labels)
            
            return pr
            
        except Exception as e:
            raise ValueError(str(e))

    def list_prs(self, state: str = "open") -> List[Dict]:
        """List pull requests"""
        url = f"{self.api_url}/repos/{self.owner}/{self.repo}/pulls"
        params = {"state": state}
        
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to list PRs: {response.json().get('message', '')}")
        
        return response.json()

class PRReviewManager:
    """Enhanced PR review manager with structured review types"""
    
    def __init__(self):
        self.github_api = GitHubAPI()
        self.dependency_manager = DependencyManager()
        self.review_templates = self._load_review_templates()
    
    def _load_review_templates(self) -> Dict[ReviewType, ReviewTemplate]:
        """Load predefined review templates"""
        templates = {}
        
        # Quick Review Template
        templates[ReviewType.QUICK] = ReviewTemplate(
            name="Quick Review",
            review_type=ReviewType.QUICK,
            checklist=[
                "Code follows project conventions",
                "Changes are focused and minimal",
                "No obvious bugs or issues",
                "Tests are included if needed"
            ],
            focus_areas=["functionality", "code_style", "tests"],
            ai_prompt_template="""Provide a quick, practical review of this PR. Focus on:
1. Overall code quality and style
2. Logic and functionality 
3. Potential issues or improvements
4. Testing adequacy

Keep it concise but thorough. Give your honest assessment."""
        )
        
        # Detailed Review Template
        templates[ReviewType.DETAILED] = ReviewTemplate(
            name="Detailed Technical Review",
            review_type=ReviewType.DETAILED,
            checklist=[
                "Architecture and design patterns",
                "Code maintainability and readability",
                "Error handling and edge cases",
                "Performance implications",
                "Security considerations",
                "Test coverage and quality",
                "Documentation updates",
                "Breaking changes identified"
            ],
            focus_areas=["architecture", "maintainability", "performance", "security", "testing"],
            ai_prompt_template="""Conduct a comprehensive technical review of this PR. Analyze:

ARCHITECTURE & DESIGN:
- Design patterns and architectural decisions
- Code organization and structure
- Separation of concerns

CODE QUALITY:
- Readability and maintainability
- Error handling and edge cases
- Performance implications
- Memory usage and optimization

SECURITY:
- Security vulnerabilities
- Input validation
- Authentication/authorization
- Data handling

TESTING:
- Test coverage adequacy
- Test quality and reliability
- Edge case coverage

Provide specific, actionable feedback with examples."""
        )
        
        # Security Review Template
        templates[ReviewType.SECURITY] = ReviewTemplate(
            name="Security Review",
            review_type=ReviewType.SECURITY,
            checklist=[
                "Input validation and sanitization",
                "Authentication and authorization",
                "Data encryption and storage",
                "SQL injection prevention",
                "XSS protection",
                "CSRF protection",
                "Secrets and sensitive data handling",
                "Access control implementation"
            ],
            focus_areas=["security", "validation", "encryption", "access_control"],
            ai_prompt_template="""Conduct a security-focused review of this PR. Examine:

SECURITY VULNERABILITIES:
- SQL injection, XSS, CSRF vulnerabilities
- Authentication and authorization flaws
- Input validation weaknesses
- Insecure data handling

DATA PROTECTION:
- Sensitive data exposure
- Encryption implementation
- Secrets management
- Access control mechanisms

BEST PRACTICES:
- Security coding standards
- OWASP compliance
- Secure configuration

Identify specific security risks and provide mitigation strategies."""
        )
        
        # Performance Review Template
        templates[ReviewType.PERFORMANCE] = ReviewTemplate(
            name="Performance Review",
            review_type=ReviewType.PERFORMANCE,
            checklist=[
                "Algorithm efficiency",
                "Database query optimization",
                "Memory usage patterns",
                "Caching strategies",
                "API response times",
                "Resource utilization",
                "Scalability considerations",
                "Performance monitoring"
            ],
            focus_areas=["performance", "scalability", "optimization", "monitoring"],
            ai_prompt_template="""Analyze this PR for performance implications:

PERFORMANCE ANALYSIS:
- Algorithm complexity and efficiency
- Database query performance
- Memory usage and leaks
- CPU-intensive operations

SCALABILITY:
- Scalability bottlenecks
- Resource utilization
- Concurrent access patterns
- Load handling capabilities

OPTIMIZATION:
- Caching opportunities
- Code optimization potential
- Database indexing needs
- Network efficiency

Provide specific performance recommendations and potential optimizations."""
        )
        
        # Dependencies Review Template
        templates[ReviewType.DEPENDENCIES] = ReviewTemplate(
            name="Dependencies Review",
            review_type=ReviewType.DEPENDENCIES,
            checklist=[
                "New dependencies justified",
                "Version compatibility checked",
                "Security vulnerabilities in deps",
                "Bundle size impact",
                "License compatibility",
                "Maintenance status of deps",
                "Alternative solutions considered",
                "Dependency conflicts resolved"
            ],
            focus_areas=["dependencies", "security", "compatibility", "licensing"],
            ai_prompt_template="""Review dependency changes in this PR:

DEPENDENCY ANALYSIS:
- New dependencies and their necessity
- Version updates and compatibility
- Security implications of changes
- Bundle size and performance impact

RISK ASSESSMENT:
- Maintenance status of dependencies
- Known vulnerabilities
- License compatibility
- Breaking change potential

RECOMMENDATIONS:
- Alternative solutions
- Version pinning strategies
- Security considerations
- Update pathways

Focus on dependency health and project impact."""
        )
        
        return templates
    
    def conduct_review(self, pr_number: int, review_type: ReviewType = ReviewType.QUICK, 
                      auto_post: bool = False, include_checklist: bool = True) -> ReviewResult:
        """Conduct a structured review of a PR"""
        
        try:
            # Get PR information
            pr = self.github_api.get_pr_by_number(pr_number)
            diff = self.github_api.get_pr_diff(pr_number)
            files = self.github_api.get_pr_files(pr_number)
            
            # Get review template
            template = self.review_templates.get(review_type)
            if not template:
                raise ValueError(f"Unknown review type: {review_type}")
            
            # Conduct specific review type
            if review_type == ReviewType.DEPENDENCIES:
                return self._conduct_dependency_review(pr, diff, files, template, auto_post)
            else:
                return self._conduct_general_review(pr, diff, files, template, auto_post, include_checklist)
                
        except Exception as e:
            console.print(Panel(
                f"[red]Review failed: {str(e)}[/red]",
                title="Review Error",
                border_style="red"
            ))
            raise
    
    def _conduct_general_review(self, pr: Dict, diff: str, files: List[Dict], 
                              template: ReviewTemplate, auto_post: bool, include_checklist: bool) -> ReviewResult:
        """Conduct a general AI-powered review"""
        
        # Prepare context
        files_changed = [f['filename'] for f in files]
        pr_context = {
            'title': pr['title'],
            'body': pr.get('body', '') or '',
            'author': pr['user']['login'],
            'files_changed': files_changed,
            'additions': pr.get('additions', 0),
            'deletions': pr.get('deletions', 0)
        }
        
        # Create AI prompt
        context = f"""PR Information:
Title: {pr_context['title']}
Author: {pr_context['author']}
Files Changed: {len(files_changed)} files
Lines: +{pr_context['additions']} -{pr_context['deletions']}

Description: {pr_context['body'][:500]}...

Files Modified:
{chr(10).join(files_changed[:10])}
{'...' if len(files_changed) > 10 else ''}

Focus Areas: {', '.join(template.focus_areas)}
"""
        
        # Truncate diff if too large
        from .ai_providers import AIProvider
        estimated_tokens = AIProvider.estimate_tokens(diff)
        
        if estimated_tokens > 60000:
            console.print(f"[yellow]Large diff detected ({estimated_tokens:,} tokens). Using intelligent analysis...[/yellow]")
            if estimated_tokens > 150000:
                diff_summary = AIProvider.get_diff_summary(diff)
                diff = f"SUMMARY OF CHANGES:\n{diff_summary}\n\nFIRST 100 LINES OF DIFF:\n" + '\n'.join(diff.split('\n')[:100])
            else:
                diff = AIProvider.truncate_diff_intelligently(diff, max_tokens=60000)
        
        full_prompt = f"""{template.ai_prompt_template}

{context}

CODE CHANGES:
{diff}

Provide your review in this format:
SUMMARY: (2-3 sentences about the changes)
ASSESSMENT: (APPROVE/REQUEST_CHANGES/COMMENT)
FEEDBACK: (detailed feedback and recommendations)
CONFIDENCE: (0.0-1.0 confidence in this review)
"""

        # Generate review
        provider = settings.get_provider()
        model = settings.get_model()
        
        try:
            if provider == 'openai':
                api_key = settings.get_openai_api_key()
                ai = OpenAIProvider(api_key)
            else:
                ai = OllamaProvider()
            
            review_text = ai.generate_commit_message(full_prompt, model)
            
        except Exception as e:
            # Fallback to basic review
            review_text = f"SUMMARY: Unable to generate detailed review due to: {str(e)}\nASSESSMENT: COMMENT\nFEEDBACK: Please review manually.\nCONFIDENCE: 0.1"
        
        # Parse review response
        summary, assessment, feedback, confidence = self._parse_review_response(review_text)
        
        # Generate checklist if requested
        checklist_items = None
        if include_checklist:
            checklist_items = self._generate_checklist_assessment(template.checklist, diff, files_changed)
        
        # Create review result
        result = ReviewResult(
            pr_number=pr['number'],
            review_type=template.review_type,
            assessment=assessment,
            summary=summary,
            detailed_feedback=feedback,
            checklist_items=checklist_items,
            confidence_score=confidence
        )
        
        # Post review if requested
        if auto_post:
            self._post_review_to_github(result, pr)
        
        return result
    
    def _conduct_dependency_review(self, pr: Dict, diff: str, files: List[Dict], 
                                 template: ReviewTemplate, auto_post: bool) -> ReviewResult:
        """Conduct a dependency-focused review"""
        
        # Analyze dependency changes
        dependency_changes = self.dependency_manager.analyze_commit_dependencies()
        detected_deps = self.dependency_manager.analyze_code_for_dependencies()
        
        # Generate dependency insights
        dep_insights = self.dependency_manager.generate_dependency_insights(dependency_changes)
        
        # Check for dependency issues
        dependency_issues = []
        security_issues = []
        
        for change in dependency_changes:
            if change.change_type in ['updated', 'added']:
                # Check for potential security issues
                if any(keyword in change.name.lower() for keyword in ['crypto', 'auth', 'security', 'ssl']):
                    security_issues.append(f"Security-related dependency {change.name} was {change.change_type}")
                
                # Check for major version updates
                if change.old_version and change.new_version:
                    breaking_changes = self.dependency_manager.detect_breaking_changes(
                        change.name, change.old_version, change.new_version, change.ecosystem
                    )
                    if breaking_changes:
                        dependency_issues.extend(breaking_changes)
        
        # Create summary
        summary = f"Dependency review: {len(dependency_changes)} changes, {len(detected_deps)} missing dependencies detected"
        
        # Generate assessment
        assessment = "COMMENT"
        if security_issues or dependency_issues:
            assessment = "REQUEST_CHANGES"
        elif dependency_changes or detected_deps:
            assessment = "COMMENT"
        else:
            assessment = "APPROVE"
        
        # Combine feedback
        feedback_parts = []
        if dependency_changes:
            feedback_parts.append(f"**Dependency Changes:** {len(dependency_changes)} detected")
        if detected_deps:
            feedback_parts.append(f"**Missing Dependencies:** {len(detected_deps)} packages need to be added")
        if security_issues:
            feedback_parts.append(f"**Security Concerns:** {len(security_issues)} issues identified")
        if dependency_issues:
            feedback_parts.append(f"**Breaking Changes:** {len(dependency_issues)} potential issues")
        
        feedback_parts.append(f"**AI Analysis:** {dep_insights}")
        
        feedback = "\n\n".join(feedback_parts)
        
        # Create result
        result = ReviewResult(
            pr_number=pr['number'],
            review_type=ReviewType.DEPENDENCIES,
            assessment=assessment,
            summary=summary,
            detailed_feedback=feedback,
            dependency_issues=dependency_issues,
            security_issues=security_issues,
            confidence_score=0.9
        )
        
        # Post review if requested
        if auto_post:
            self._post_review_to_github(result, pr)
        
        return result
    
    def _parse_review_response(self, review_text: str) -> tuple:
        """Parse AI review response into components"""
        
        summary = "Review completed"
        assessment = "COMMENT"
        feedback = review_text
        confidence = 0.7
        
        try:
            lines = review_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('SUMMARY:'):
                    summary = line.replace('SUMMARY:', '').strip()
                elif line.startswith('ASSESSMENT:'):
                    assessment_text = line.replace('ASSESSMENT:', '').strip().upper()
                    if assessment_text in ['APPROVE', 'REQUEST_CHANGES', 'COMMENT']:
                        assessment = assessment_text
                elif line.startswith('FEEDBACK:'):
                    feedback = line.replace('FEEDBACK:', '').strip()
                    # Include remaining lines as feedback
                    remaining_lines = lines[lines.index(line)+1:]
                    if remaining_lines:
                        feedback += '\n' + '\n'.join(remaining_lines)
                    break
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.replace('CONFIDENCE:', '').strip())
                    except:
                        pass
        except:
            pass
        
        return summary, assessment, feedback, confidence
    
    def _generate_checklist_assessment(self, checklist: List[str], diff: str, files_changed: List[str]) -> List[Dict]:
        """Generate checklist assessment based on code changes"""
        
        checklist_items = []
        
        for item in checklist:
            status = "unknown"
            notes = ""
            
            # Simple heuristics for checklist assessment
            if "test" in item.lower():
                if any('test' in f.lower() for f in files_changed):
                    status = "passed"
                    notes = "Test files modified"
                else:
                    status = "attention"
                    notes = "No test file changes detected"
            
            elif "documentation" in item.lower() or "readme" in item.lower():
                if any(f.lower().endswith(('.md', '.rst', '.txt')) for f in files_changed):
                    status = "passed"
                    notes = "Documentation files updated"
                else:
                    status = "attention"
                    notes = "Consider updating documentation"
            
            elif "security" in item.lower():
                if any(keyword in diff.lower() for keyword in ['password', 'token', 'secret', 'api_key', 'auth']):
                    status = "attention"
                    notes = "Security-related code detected"
                else:
                    status = "passed"
                    notes = "No obvious security concerns"
            
            else:
                status = "review_needed"
                notes = "Manual review required"
            
            checklist_items.append({
                'item': item,
                'status': status,
                'notes': notes
            })
        
        return checklist_items
    
    def _post_review_to_github(self, result: ReviewResult, pr: Dict):
        """Post review result to GitHub"""
        
        # Format review body
        review_body_parts = [
            f"## 🤖 {result.review_type.value.title()} Review",
            "",
            f"**Summary:** {result.summary}",
            "",
            result.detailed_feedback
        ]
        
        # Add checklist if available
        if result.checklist_items:
            review_body_parts.extend([
                "",
                "## 📋 Review Checklist",
                ""
            ])
            
            for item in result.checklist_items:
                status_icon = {
                    'passed': '✅',
                    'attention': '⚠️',
                    'failed': '❌',
                    'review_needed': '👀',
                    'unknown': '❓'
                }.get(item['status'], '❓')
                
                review_body_parts.append(f"- {status_icon} {item['item']}")
                if item['notes']:
                    review_body_parts.append(f"  - {item['notes']}")
        
        # Add confidence score
        review_body_parts.extend([
            "",
            f"*AI Confidence: {result.confidence_score:.0%}*"
        ])
        
        review_body = '\n'.join(review_body_parts)
        
        try:
            self.github_api.post_pr_review(
                result.pr_number,
                review_body,
                event=result.assessment
            )
            console.print(f"[green]✅ Review posted to PR #{result.pr_number}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to post review: {str(e)}[/red]")
    
    def bulk_review(self, review_type: ReviewType = ReviewType.QUICK, 
                   max_prs: int = 5, auto_post: bool = False) -> List[ReviewResult]:
        """Review multiple open PRs"""
        
        try:
            prs = self.github_api.list_prs(state="open")
            if not prs:
                console.print("[yellow]No open PRs found[/yellow]")
                return []
            
            # Limit PRs to review
            prs_to_review = prs[:max_prs]
            results = []
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                for pr in prs_to_review:
                    task = progress.add_task(f"Reviewing PR #{pr['number']}...", total=None)
                    
                    try:
                        result = self.conduct_review(
                            pr['number'], 
                            review_type=review_type,
                            auto_post=auto_post,
                            include_checklist=False  # Skip checklist for bulk reviews
                        )
                        results.append(result)
                        progress.update(task, completed=True)
                        
                    except Exception as e:
                        progress.update(task, visible=False)
                        console.print(f"[yellow]Failed to review PR #{pr['number']}: {str(e)}[/yellow]")
            
            return results
            
        except Exception as e:
            console.print(Panel(
                f"[red]Bulk review failed: {str(e)}[/red]",
                title="Error",
                border_style="red"
            ))
            return []
    
    def compare_reviews(self, pr_number: int, review_types: List[ReviewType]) -> Dict[ReviewType, ReviewResult]:
        """Compare multiple review types for the same PR"""
        
        results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for review_type in review_types:
                task = progress.add_task(f"Conducting {review_type.value} review...", total=None)
                
                try:
                    result = self.conduct_review(
                        pr_number, 
                        review_type=review_type,
                        auto_post=False,
                        include_checklist=False
                    )
                    results[review_type] = result
                    progress.update(task, completed=True)
                    
                except Exception as e:
                    progress.update(task, visible=False)
                    console.print(f"[yellow]Failed {review_type.value} review: {str(e)}[/yellow]")
        
        return results

def infer_labels_from_content(title: str, body: str) -> List[str]:
    """Infer appropriate labels from PR title and body"""
    labels = set()
    
    # Common keywords mapping
    keyword_to_label = {
        'feat': 'feature',
        'fix': 'bug',
        'doc': 'documentation',
        'style': 'enhancement',
        'refactor': 'refactor',
        'perf': 'performance',
        'test': 'testing',
        'chore': 'maintenance',
        'add': 'feature',
        'improve': 'enhancement',
        'update': 'enhancement',
        'optimize': 'performance',
        'bugfix': 'bug',
        'error': 'bug',
        'issue': 'bug',
        'document': 'documentation',
        'readme': 'documentation',
    }
    
    # Check conventional commit format in title
    if '(' in title and '):' in title:
        commit_type = title.split('(')[0].lower()
        if commit_type in keyword_to_label:
            labels.add(keyword_to_label[commit_type])
    
    # Check keywords in title and body
    content = (title + ' ' + body).lower()
    for keyword, label in keyword_to_label.items():
        if keyword in content:
            labels.add(label)
    
    # Always return at least one label
    if not labels:
        labels.add('enhancement')
    
    return list(labels)

def parse_ai_response(response: str) -> Dict[str, str]:
    """Parse AI response with robust error handling"""
    try:
        # Extract title
        title_parts = response.split('<title>')
        if len(title_parts) < 2:
            # Try to get first line as title
            lines = response.strip().split('\n')
            title = lines[0].strip()
            if not title:
                raise ValueError("Could not find title section")
        else:
            title = title_parts[1].split('</title>')[0].strip()
        
        # Extract body
        body_parts = response.split('<body>')
        if len(body_parts) < 2:
            # Try to extract markdown sections as body
            body_lines = []
            in_section = False
            for line in response.split('\n'):
                if line.startswith('##'):
                    in_section = True
                if in_section:
                    body_lines.append(line)
            body = '\n'.join(body_lines).strip()
            if not body:
                body = "## Changes\n" + response.strip()  # Use full response as changes
        else:
            body = body_parts[1].split('</body>')[0].strip()
        
        # Extract or infer labels
        labels = []
        labels_parts = response.split('<labels>')
        if len(labels_parts) >= 2:
            labels_text = labels_parts[1].split('</labels>')[0].strip()
            labels = [l.strip() for l in labels_text.split(',') if l.strip()]
        
        # If no labels found or invalid, infer from content
        if not labels:
            labels = infer_labels_from_content(title, body)
        
        # Validate parsed content
        if not title:
            raise ValueError("Empty title")
        if not body:
            raise ValueError("Empty body")
        
        return {
            'title': title,
            'body': body,
            'labels': labels
        }
    except Exception as e:
        raise ValueError(f"Failed to parse AI response: {str(e)}\nResponse:\n{response}")

def generate_pr_content(branch: str = None, progress: Progress = None) -> Dict[str, str]:
    """Generate PR title, body, and labels using AI"""
    if not branch:
        branch = get_current_branch()
    
    task = None
    if progress:
        task = progress.add_task("Analyzing changes...", total=None)
    
    try:
        # Get base branch
        base_branch = get_default_branch()
        
        # Get changes
        merge_base = run_git_command(['merge-base', base_branch, branch]).stdout.strip()
        diff = run_git_command(['diff', merge_base, branch]).stdout
        commits = run_git_command(['log', '--oneline', f'{merge_base}..{branch}']).stdout
        
        if not commits.strip():
            if task:
                progress.update(task, visible=False)
            console.print(Panel(
                "[red]No commits found in this branch[/red]\n\n"
                "[yellow]To create a PR, you need to:[/yellow]\n"
                "1. Create a branch (if not done):\n"
                "   [blue]sl branch \"your feature description\"[/blue]\n"
                "2. Make your changes and commit them:\n"
                "   [blue]sl g -a[/blue] (auto-stages and commits changes)\n"
                "3. Then create the PR:\n"
                "   [blue]sl pr create --details[/blue]",
                title="Workflow Guide",
                border_style="yellow"
            ))
            sys.exit(1)
        
        # Optimize diff for AI processing
        from .ai_providers import AIProvider
        estimated_tokens = AIProvider.estimate_tokens(diff)
        
        if estimated_tokens > 60000:
            console.print(f"[yellow]Large diff detected ({estimated_tokens:,} tokens). Using intelligent analysis...[/yellow]")
            if estimated_tokens > 150000:
                diff_summary = AIProvider.get_diff_summary(diff)
                diff = f"SUMMARY OF CHANGES:\n{diff_summary}\n\nFIRST 100 LINES OF DIFF:\n" + '\n'.join(diff.split('\n')[:100])
            else:
                diff = AIProvider.truncate_diff_intelligently(diff, max_tokens=60000)
        
        provider = settings.get_provider()
        model = settings.get_model()
        
        # First, generate a high-level understanding of the changes
        analysis_prompt = f"""Analyze these changes and provide a high-level understanding:

Commits:
{commits}

Changes:
{diff}

Respond in this format:
1. Main purpose (one line):
2. Type of changes (feat/fix/refactor/etc):
3. Scope of changes:
4. Breaking changes (yes/no):
5. Key files/components affected:"""

        try:
            if provider == 'openai':
                api_key = settings.get_openai_api_key()
                ai = OpenAIProvider(api_key)
            else:
                ai = OllamaProvider()
            
            analysis = ai.generate_commit_message(analysis_prompt, model)
            
            # Extract key information from analysis
            change_type = 'feat'  # default
            if 'Type of changes:' in analysis:
                type_line = analysis.split('Type of changes:')[1].split('\n')[0].strip().lower()
                if any(t in type_line for t in ['feat', 'fix', 'docs', 'style', 'refactor', 'perf', 'test', 'chore']):
                    for t in ['feat', 'fix', 'docs', 'style', 'refactor', 'perf', 'test', 'chore']:
                        if t in type_line:
                            change_type = t
                            break
            
            main_purpose = ""
            if 'Main purpose:' in analysis:
                main_purpose = analysis.split('Main purpose:')[1].split('\n')[0].strip()
            
            scope = ""
            if 'Scope of changes:' in analysis:
                scope = analysis.split('Scope of changes:')[1].split('\n')[0].strip()
            
            is_breaking = False
            if 'Breaking changes:' in analysis:
                is_breaking = 'yes' in analysis.split('Breaking changes:')[1].split('\n')[0].strip().lower()
            
            # Create the format string for the title format requirement
            title_format = f"{change_type}({scope if scope else 'scope'}): description"
            
            # Now use this analysis to generate the PR content
            pr_prompt = f"""Based on this analysis of the changes, generate a pull request:

Analysis:
{analysis}

Commits:
{commits}

Changes:
{diff}

Requirements:
1. Title must be clear and follow conventional commit format: {title_format}
2. Description must be detailed but concise
3. Include specific testing instructions
4. {"Highlight breaking changes and" if is_breaking else "Include any"} dependencies
5. Focus on what reviewers need to know

Format the response EXACTLY like this:

Title: {change_type}(scope): brief description

## Overview
{main_purpose or "(2-3 sentences about the main changes and their purpose)"}

## Changes
- (bullet points of specific changes)
- (focus on what and why, not how)

## Testing
- (specific steps to test the changes)
- (expected outcomes)
- (any test data or setup needed)

## Notes
- Breaking Changes: {"Yes - " if is_breaking else "None"}
- Dependencies: (if any)
- Migration: (if needed)

Labels: feature, bug, documentation, enhancement, refactor, performance, testing, maintenance (comma-separated, choose relevant ones only)"""

            response = ai.generate_commit_message(pr_prompt, model)
            
            # Parse response with improved error handling
            try:
                # Extract title
                title = None
                body_lines = []
                labels = []
                
                lines = [l for l in response.split('\n') if l.strip()]
                
                # Find title
                for line in lines:
                    if line.startswith('Title:'):
                        title = line.replace('Title:', '').strip()
                        break
                    elif any(line.startswith(f"{t}(") for t in ['feat', 'fix', 'docs', 'style', 'refactor', 'perf', 'test', 'chore']):
                        title = line.strip()
                        break
                
                # If no title found, construct one from analysis
                if not title:
                    scope_text = scope if scope else "general"
                    title = f"{change_type}({scope_text}): {main_purpose or 'update codebase'}"
                
                # Extract body and labels
                in_body = False
                for line in lines:
                    line = line.strip()
                    if line.lower().startswith('labels:'):
                        # Extract labels
                        labels_text = line.split(':', 1)[1].strip()
                        labels = [l.strip() for l in labels_text.split(',') if l.strip()]
                        break
                    elif line.startswith('##') or in_body:
                        in_body = True
                        body_lines.append(line)
                
                # Clean up body
                body = '\n'.join(body_lines).strip()
                if not body:
                    # Construct a basic body from analysis
                    body = '\n'.join([
                        "## Overview",
                        main_purpose or "Updates to the codebase",
                        "",
                        "## Changes",
                        "- " + "\n- ".join(commits.split('\n')[:5]),
                        "",
                        "## Testing",
                        "- Please verify the changes work as expected"
                    ])
                    if is_breaking:
                        body += "\n\n## Notes\n- Breaking Changes: Yes - please review carefully"
                
                # If no labels found or invalid, infer from content
                if not labels:
                    labels = infer_labels_from_content(title, body)
                
                if task:
                    progress.update(task, completed=True)
                
                return {
                    'title': title,
                    'body': body,
                    'labels': labels
                }
            except Exception as e:
                # If parsing fails, create a basic PR from analysis
                if task:
                    progress.update(task, visible=False)
                scope_text = scope if scope else "general"
                title = f"{change_type}({scope_text}): {main_purpose or 'update codebase'}"
                
                # Create body using list join instead of f-string with newlines
                body_parts = [
                    "## Overview",
                    main_purpose or "Updates to the codebase",
                    "",
                    "## Changes",
                    "- " + "\n- ".join(commits.split('\n')[:5]),
                    "",
                    "## Testing",
                    "- Please verify the changes work as expected"
                ]
                
                if is_breaking:
                    body_parts.extend(["", "## Notes", "- Breaking Changes: Yes - please review carefully"])
                
                body = '\n'.join(body_parts)
                labels = infer_labels_from_content(title, body)
                
                return {
                    'title': title,
                    'body': body,
                    'labels': labels
                }
            
        except Exception as e:
            if task:
                progress.update(task, visible=False)
            raise ValueError(f"Failed to generate PR content: {str(e)}")
            
    except Exception as e:
        if task:
            progress.update(task, visible=False)
        raise e

def push_branch(branch: str) -> bool:
    """Push branch to GitHub"""
    try:
        console.print("[yellow]Branch not found on GitHub. Pushing changes...[/yellow]")
        result = run_git_command(['push', '-u', 'origin', branch], check=False)
        if result.returncode == 0:
            console.print("[green]✓[/green] Successfully pushed branch to GitHub")
            return True
        else:
            error = result.stderr.decode('utf-8').strip()
            if "remote: Repository not found" in error:
                raise ValueError(
                    "Repository not found on GitHub. Please ensure:\n"
                    "1. The repository exists on GitHub\n"
                    "2. You have write access to the repository\n"
                    "3. Your GitHub token has the correct permissions"
                )
            elif "Permission denied" in error:
                raise ValueError(
                    "Permission denied. Please ensure:\n"
                    "1. You have write access to the repository\n"
                    "2. Your GitHub token has the correct permissions\n"
                    "3. The repository URL is correct"
                )
            else:
                raise ValueError(f"Failed to push branch: {error}")
    except Exception as e:
        raise ValueError(str(e))

def create_pr(base: str = None, show_details: bool = False, auto_push: bool = True) -> None:
    """Create a pull request with AI-generated content"""
    github = GitHubAPI()
    content = None
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        try:
            # Generate PR content
            content = generate_pr_content(progress=progress)
            
            # Track PR content generation
            from .usage_tracker import track_command_manual
            track_command_manual(
                command="pr",
                subcommand="content_generation",
                success=True,
                input_data=f"Branch: {get_current_branch()}",
                output_data=f"Title: {content['title']}\nBody: {content['body']}\nLabels: {content['labels']}",
                input_type="git_context",
                parameters={"provider": settings.get_provider(), "action": "create"}
            )
        except Exception as e:
            console.print(Panel(f"[red]Failed to generate PR content: {str(e)}[/red]", title="Error", border_style="red"))
            sys.exit(1)
    
    # Show preview outside of progress context
    console.print("\n[bold]Pull Request Preview[/bold]")
    console.print(Panel(
        "\n".join([
            f"[bold cyan]Title:[/bold cyan] {content['title']}",
            "",
            f"[bold cyan]Body:[/bold cyan]",
            content['body'],
            "",
            f"[bold cyan]Labels:[/bold cyan] {', '.join(content['labels'])}",
        ]),
        title="Preview",
        border_style="cyan",
        padding=(1, 2)
    ))
    
    # Confirm creation
    if not typer.confirm("\n💭 Create this pull request?", default=True):
        console.print("\n[yellow]PR creation cancelled[/yellow]")
        return
    
    # Create PR in a new progress context
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task_create = progress.add_task("Creating pull request...", total=None)
        try:
            head = get_current_branch()
            
            # Try to validate PR parameters
            default_base = base or get_default_branch()
            try:
                github.validate_pr_params(head, default_base)
            except ValueError as e:
                if "Bad credentials" in str(e) or "401" in str(e):
                    console.print("[yellow]⚠️  Skipping validation due to GitHub authentication issue[/yellow]")
                    console.print("[yellow]   Proceeding with PR creation...[/yellow]")
                elif auto_push and "not found on GitHub" in str(e):
                    # Try to push the branch
                    try:
                        if push_branch(head):
                            # Retry validation after successful push
                            try:
                                github.validate_pr_params(head, default_base)
                            except ValueError as retry_e:
                                if "Bad credentials" in str(retry_e) or "401" in str(retry_e):
                                    console.print("[yellow]⚠️  Validation failed due to auth, but branch was pushed[/yellow]")
                                else:
                                    raise retry_e
                        else:
                            raise ValueError("Failed to push branch")
                    except Exception as push_error:
                        progress.update(task_create, visible=False)
                        console.print(Panel(f"[red]{str(push_error)}[/red]", title="Error", border_style="red"))
                        sys.exit(1)
                else:
                    raise e
            
            # Create PR
            pr = github.create_pr(
                title=content['title'],
                body=content['body'],
                base=default_base,
                labels=content['labels']
            )
            progress.update(task_create, completed=True)
            
            # Show success message with PR link
            console.print(Panel(
                f"[green]✓ Pull request created successfully![/green]\n\n"
                f"View PR: [link={pr['html_url']}]{pr['html_url']}[/link]",
                title="Success",
                border_style="green",
                padding=(1, 2)
            ))
            
        except Exception as e:
            progress.update(task_create, visible=False)
            console.print(Panel(
                f"[red]Failed to create PR: {str(e)}[/red]",
                title="Error",
                border_style="red"
            ))
            sys.exit(1)

def list_prs(show_details: bool = False) -> None:
    """List pull requests with optional AI insights"""
    github_api = GitHubAPI()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Fetching pull requests...", total=None)
        
        try:
            prs = github_api.list_prs()
            progress.update(task, completed=True)
            
            if not prs:
                console.print(Panel(
                    "[yellow]No open pull requests found[/yellow]",
                    title="Pull Requests",
                    border_style="yellow"
                ))
                return
            
            table = Table(title="Open Pull Requests", show_header=True, header_style="bold cyan")
            table.add_column("#", style="cyan", width=6)
            table.add_column("Title", style="white", min_width=30)
            table.add_column("Author", style="green")
            table.add_column("Branch", style="yellow")
            if show_details:
                table.add_column("AI Insights", style="blue", min_width=40)
            
            for pr in prs[:10]:  # Limit to 10 PRs
                row = [
                    str(pr['number']),
                    pr['title'][:50] + ('...' if len(pr['title']) > 50 else ''),
                    pr['user']['login'],
                    pr['head']['ref']
                ]
                
                if show_details:
                    # Generate AI insights for the PR
                    task_insight = progress.add_task(f"Analyzing PR #{pr['number']}...", total=None)
                    try:
                        insights = generate_pr_insights(pr)
                        progress.update(task_insight, completed=True)
                        row.append(insights)
                    except Exception:
                        progress.update(task_insight, visible=False)
                        row.append("[dim]Unable to generate insights[/dim]")
                
                table.add_row(*row)
            
            console.print(table)
            
        except Exception as e:
            progress.update(task, visible=False)
            console.print(Panel(
                f"[red]Failed to list pull requests: {str(e)}[/red]",
                title="Error",
                border_style="red"
            ))

def generate_pr_insights(pr: Dict) -> str:
    """Generate quick AI insights about a PR"""
    try:
        provider = settings.get_provider()
        model = settings.get_model()
        
        # Get basic PR info
        title = pr['title']
        body = pr.get('body', '') or ''
        
        prompt = f"""Provide a one-line insight about this pull request:

Title: {title}
Description: {body[:200]}...

Focus on the main purpose and potential impact. Keep it under 50 characters."""

        if provider == 'openai':
            api_key = settings.get_openai_api_key()
            ai = OpenAIProvider(api_key)
        else:
            ai = OllamaProvider()
        
        insight = ai.generate_commit_message(prompt, model).strip()
        return insight[:50] + ('...' if len(insight) > 50 else '')
    except Exception:
        return "No insights available"

def generate_code_review(diff: str, files_changed: List[str] = None, pr_context: Dict = None) -> Dict[str, str]:
    """Generate AI-powered code review"""
    provider = settings.get_provider()
    model = settings.get_model()
    
    # Create context information
    context = ""
    if pr_context:
        context += f"PR Title: {pr_context.get('title', '')}\n"
        context += f"PR Description: {pr_context.get('body', '')[:300]}...\n\n"
    
    if files_changed:
        context += f"Files Changed: {', '.join(files_changed[:10])}\n\n"
    
    # Truncate diff if too large (using the existing logic from ai_providers)
    from .ai_providers import AIProvider
    estimated_tokens = AIProvider.estimate_tokens(diff)
    
    if estimated_tokens > 60000:  # Leave room for prompt and response
        console.print(f"[yellow]Large diff detected ({estimated_tokens:,} estimated tokens). Using intelligent analysis...[/yellow]")
        if estimated_tokens > 150000:
            diff_summary = AIProvider.get_diff_summary(diff)
            diff = f"SUMMARY OF CHANGES:\n{diff_summary}\n\nFIRST 100 LINES OF DIFF:\n" + '\n'.join(diff.split('\n')[:100])
        else:
            diff = AIProvider.truncate_diff_intelligently(diff, max_tokens=60000)
    
    prompt = f"""You are reviewing a teammate's pull request. Give me your honest thoughts in exactly one conversational sentence.

CRITICAL: Do NOT repeat the commit message or PR title. Do NOT use any formatting. Write like you're casually talking to a colleague.

{context}

CODE CHANGES:
{diff}

You are a colleague doing a casual + thorough code review. First, explain thoroughly what was implemented, what was done, that sort of stuff. Then you can proceed to explain the codechanges, Respond with  natural VERY short pargraphs  giving your honest thoughts about the code changes made, Once was made basically catching someone up to date, like what was done and why, Jin just made the connotations of potential logics, but I want you to be very dynamic in the way you write this, think and reason properly and naturally. . DO NOT repeat commit messages or use any formatting. Talk like you're sitting next to a teammate.
Remember, keep it concise so you can read it with a couple of glances. 
WRITE IN A CONVERSATIONAL LANGUAGE, LIKE YOU'RE SITTING NEXT TO A TEAMMATE. AND WRITE WITH INSANE CLARITY 
Your casual review:"""

    try:
        if provider == 'openai':
            api_key = settings.get_openai_api_key()
            ai = OpenAIProvider(api_key)
        else:
            ai = OllamaProvider()
        
        # Generate the review
        review = ai.generate_commit_message(prompt, model)
        
        # Check if review was generated
        if not review or not review.strip():
            raise Exception(f"AI provider {provider} returned empty response")
        
        # For single sentence reviews, determine assessment from sentiment
        assessment = "COMMENT"  # default for single sentence reviews
        
        # Simple sentiment analysis for assessment
        review_lower = review.lower()
        if any(word in review_lower for word in ['looks good', 'solid', 'clean', 'confident', 'nice', 'well done', 'great']):
            if not any(word in review_lower for word in ['but', 'however', 'though', 'issue', 'problem', 'concern']):
                assessment = "APPROVE"
        elif any(word in review_lower for word in ['issue', 'problem', 'concern', 'fix', 'before merge', 'needs']):
            assessment = "REQUEST_CHANGES"
        
        # Clean up the review (remove any extra formatting)
        clean_review = review.strip()
        if clean_review.startswith('"') and clean_review.endswith('"'):
            clean_review = clean_review[1:-1]
        
        # Track the code review generation
        from .usage_tracker import track_command_manual
        track_command_manual(
            command="review",
            subcommand="code_review_generation",
            success=True,
            input_data=diff[:1000] + "..." if len(diff) > 1000 else diff,  # Truncate for storage
            output_data=clean_review,
            input_type="code_diff",
            parameters={"provider": provider, "assessment": assessment, "files_count": len(files_changed) if files_changed else 0}
        )
        
        return {
            'review': clean_review,
            'assessment': assessment
        }
        
    except Exception as e:
        error_msg = str(e)
        
        # Try fallback to Ollama if OpenAI fails
        if provider == 'openai' and ("token" in error_msg.lower() or "limit" in error_msg.lower() or "context" in error_msg.lower()):
            try:
                fallback_ai = OllamaProvider()
                review = fallback_ai.generate_commit_message(prompt, "llama2")
                if review and review.strip():
                    return {
                        'review': review,
                        'assessment': "COMMENT"
                    }
                else:
                    raise Exception("Ollama also returned empty response")
            except Exception as fallback_error:
                raise Exception(f"Both AI providers failed. OpenAI: {error_msg}, Ollama: {str(fallback_error)}")
        
        raise Exception(f"Review generation failed with {provider}: {error_msg}")

def review_current_branch(auto_comment: bool = False) -> None:
    """Review changes in the current branch"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        try:
            # Check if we're in a PR branch
            github_api = GitHubAPI()
            current_pr = github_api.get_current_pr()
            
            if current_pr and auto_comment:
                console.print(f"[cyan]Found existing PR #{current_pr['number']} for this branch[/cyan]")
                review_pr(current_pr['number'], auto_comment=True)
                return
            
            # Get branch changes
            task = progress.add_task("Getting branch changes...", total=None)
            current_branch = get_current_branch()
            
            # Find base branch
            base_branch = get_default_branch()
            
            # Get changes between base and current branch
            try:
                merge_base = run_git_command(['merge-base', base_branch, current_branch]).stdout.strip()
                diff = run_git_command(['diff', merge_base, current_branch]).stdout
                files_changed = run_git_command(['diff', '--name-only', merge_base, current_branch]).stdout.strip().split('\n')
                files_changed = [f for f in files_changed if f.strip()]
            except:
                # Fallback to staged changes if branch comparison fails
                diff = run_git_command(['diff', '--cached']).stdout
                files_changed = run_git_command(['diff', '--cached', '--name-only']).stdout.strip().split('\n')
                files_changed = [f for f in files_changed if f.strip()]
            
            if not diff.strip():
                progress.update(task, visible=False)
                console.print(Panel(
                    "[yellow]No changes found to review[/yellow]\n\n"
                    "[blue]Try one of these:[/blue]\n"
                    "• Make some changes and stage them\n"
                    "• Switch to a branch with changes\n"
                    "• Review a specific PR: [cyan]sayless review --pr 123[/cyan]",
                    title="No Changes",
                    border_style="yellow"
                ))
                return
            
            progress.update(task, completed=True)
            
            # Generate review
            task_review = progress.add_task("Generating AI code review...", total=None)
            review_result = generate_code_review(diff, files_changed)
            progress.update(task_review, completed=True)
            
            # Display review
            console.print(Panel(
                review_result['review'],
                title=f"[cyan]Code Review - {current_branch}[/cyan]",
                border_style="cyan",
                padding=(1, 2)
            ))
            
            if current_pr and not auto_comment:
                console.print(f"\n[blue]💡 This branch has PR #{current_pr['number']}. To post this review:[/blue]")
                console.print(f"[blue]   sayless review --pr {current_pr['number']} --auto-comment[/blue]")
            
        except Exception as e:
            for task_id in progress.task_ids:
                progress.update(task_id, visible=False)
            console.print(Panel(
                f"[red]Failed to review branch: {str(e)}[/red]",
                title="Error",
                border_style="red"
            ))

def review_pr(pr_number: int, auto_comment: bool = False) -> None:
    """Review a specific PR"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        try:
            github_api = GitHubAPI()
            
            # Get PR info
            task = progress.add_task(f"Getting PR #{pr_number} details...", total=None)
            pr = github_api.get_pr_by_number(pr_number)
            progress.update(task, completed=True)
            
            # Get PR diff and files
            task_diff = progress.add_task("Getting PR changes...", total=None)
            diff = github_api.get_pr_diff(pr_number)
            files = github_api.get_pr_files(pr_number)
            files_changed = [f['filename'] for f in files]
            progress.update(task_diff, completed=True)
            
            # Generate review
            task_review = progress.add_task("Generating AI code review...", total=None)
            pr_context = {
                'title': pr['title'],
                'body': pr.get('body', '') or '',
                'author': pr['user']['login']
            }
            
            try:
                review_result = generate_code_review(diff, files_changed, pr_context)
                progress.update(task_review, completed=True)
                
                # Debug: Check if review is complete
                if not review_result or not review_result.get('review'):
                    raise Exception("AI generated empty review")
                
                review_text = review_result['review']
                
                # Display review
                console.print(Panel(
                    f"**PR #{pr_number}: {pr['title']}**\n"
                    f"Author: {pr['user']['login']}\n"
                    f"Branch: `{pr['head']['ref']}` → `{pr['base']['ref']}`\n\n"
                    f"{review_text}",
                    title="[cyan]AI Code Review[/cyan]",
                    border_style="cyan",
                    padding=(1, 2)
                ))
                
            except Exception as review_error:
                progress.update(task_review, visible=False)
                
                # Fallback: show basic PR info and a simple analysis
                console.print(Panel(
                    f"**PR #{pr_number}: {pr['title']}**\n"
                    f"Author: {pr['user']['login']}\n"
                    f"Branch: `{pr['head']['ref']}` → `{pr['base']['ref']}`\n\n"
                    f"## Simple Analysis\n"
                    f"This PR modifies {len(files_changed)} files:\n"
                    f"• {', '.join(files_changed[:5])}\n"
                    f"{'• ...' if len(files_changed) > 5 else ''}\n\n"
                    f"AI review generation failed. Try again or check your AI provider configuration.",
                    title="[yellow]Basic PR Info[/yellow]",
                    border_style="yellow",
                    padding=(1, 2)
                ))
                return
            
            if auto_comment:
                # Post review to GitHub
                task_post = progress.add_task("Posting review to GitHub...", total=None)
                try:
                    review_body = f"## 🤖 AI Code Review\n\n{review_result['review']}"
                    github_api.post_pr_review(
                        pr_number, 
                        review_body, 
                        event=review_result['assessment']
                    )
                    progress.update(task_post, completed=True)
                    console.print(f"\n[green]✅ Review posted to PR #{pr_number}![/green]")
                    console.print(f"[blue]View at: {pr['html_url']}[/blue]")
                except Exception as e:
                    progress.update(task_post, visible=False)
                    console.print(f"\n[red]Failed to post review: {str(e)}[/red]")
            else:
                console.print(f"\n[blue]💡 To post this review to GitHub:[/blue]")
                console.print(f"[blue]   sayless review --pr {pr_number} --auto-comment[/blue]")
            
        except Exception as e:
            for task_id in progress.task_ids:
                progress.update(task_id, visible=False)
            console.print(Panel(
                f"[red]Failed to review PR: {str(e)}[/red]",
                title="Error",
                border_style="red"
            ))

def review_with_auto_comment() -> None:
    """Review current branch and auto-post if PR exists"""
    try:
        github_api = GitHubAPI()
        current_pr = github_api.get_current_pr()
        
        if current_pr:
            console.print(f"[cyan]Found PR #{current_pr['number']} for current branch[/cyan]")
            review_pr(current_pr['number'], auto_comment=True)
        else:
            console.print(Panel(
                "[yellow]No PR found for current branch[/yellow]\n\n"
                "[blue]Options:[/blue]\n"
                "• Create a PR first: [cyan]sayless pr create[/cyan]\n"
                "• Review without posting: [cyan]sayless review[/cyan]\n"
                "• Review specific PR: [cyan]sayless review --pr 123 --auto-comment[/cyan]",
                title="No PR Found",
                border_style="yellow"
            ))
    except Exception as e:
                    console.print(Panel(
                f"[red]Failed to find PR: {str(e)}[/red]",
                title="Error",
                border_style="red"
            )) 

# Enhanced PR Review Functions
def enhanced_review_pr(pr_number: int, review_type: str = "quick", auto_post: bool = False, include_checklist: bool = True) -> None:
    """Enhanced PR review with multiple review types"""
    
    # Convert string to ReviewType enum
    try:
        review_type_enum = ReviewType(review_type.lower())
    except ValueError:
        available_types = [rt.value for rt in ReviewType]
        console.print(Panel(
            f"[red]Invalid review type: {review_type}[/red]\n\n"
            f"Available types: {', '.join(available_types)}",
            title="Error",
            border_style="red"
        ))
        return
    
    # Create review manager and conduct review
    review_manager = PRReviewManager()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Conducting {review_type} review for PR #{pr_number}...", total=None)
        
        try:
            result = review_manager.conduct_review(
                pr_number=pr_number,
                review_type=review_type_enum,
                auto_post=auto_post,
                include_checklist=include_checklist
            )
            progress.update(task, completed=True)
            
            # Display results
            display_review_result(result)
            
            if not auto_post:
                console.print(f"\n[blue]💡 To post this review to GitHub:[/blue]")
                console.print(f"[blue]   sayless review-enhanced --pr {pr_number} --type {review_type} --auto-post[/blue]")
                
        except Exception as e:
            progress.update(task, visible=False)
            console.print(Panel(
                f"[red]Review failed: {str(e)}[/red]",
                title="Error",
                border_style="red"
            ))

def bulk_review_prs(review_type: str = "quick", max_prs: int = 5, auto_post: bool = False) -> None:
    """Review multiple PRs at once"""
    
    # Convert string to ReviewType enum
    try:
        review_type_enum = ReviewType(review_type.lower())
    except ValueError:
        available_types = [rt.value for rt in ReviewType]
        console.print(Panel(
            f"[red]Invalid review type: {review_type}[/red]\n\n"
            f"Available types: {', '.join(available_types)}",
            title="Error",
            border_style="red"
        ))
        return
    
    # Create review manager and conduct bulk review
    review_manager = PRReviewManager()
    
    console.print(Panel(
        f"[cyan]Starting bulk review of up to {max_prs} open PRs[/cyan]\n"
        f"Review Type: {review_type_enum.value.title()}\n"
        f"Auto-post: {'Yes' if auto_post else 'No'}",
        title="Bulk Review",
        border_style="cyan"
    ))
    
    results = review_manager.bulk_review(
        review_type=review_type_enum,
        max_prs=max_prs,
        auto_post=auto_post
    )
    
    if results:
        # Display summary table
        table = Table(title="Bulk Review Results", show_header=True, header_style="bold cyan")
        table.add_column("PR #", style="cyan", width=6)
        table.add_column("Assessment", style="white", width=12)
        table.add_column("Summary", style="green", min_width=40)
        table.add_column("Confidence", style="yellow", width=10)
        
        for result in results:
            assessment_color = {
                'APPROVE': '[green]APPROVE[/green]',
                'REQUEST_CHANGES': '[red]REQUEST_CHANGES[/red]',
                'COMMENT': '[yellow]COMMENT[/yellow]'
            }.get(result.assessment, result.assessment)
            
            table.add_row(
                str(result.pr_number),
                assessment_color,
                result.summary[:60] + ('...' if len(result.summary) > 60 else ''),
                f"{result.confidence_score:.0%}"
            )
        
        console.print(table)
        
        if auto_post:
            console.print(f"\n[green]✅ Posted {len(results)} reviews to GitHub[/green]")
        else:
            console.print(f"\n[blue]💡 To post these reviews, add --auto-post flag[/blue]")
    else:
        console.print("[yellow]No PRs reviewed[/yellow]")

def compare_review_types(pr_number: int) -> None:
    """Compare multiple review types for the same PR"""
    
    review_manager = PRReviewManager()
    review_types = [ReviewType.QUICK, ReviewType.DETAILED, ReviewType.SECURITY, ReviewType.PERFORMANCE]
    
    console.print(Panel(
        f"[cyan]Comparing multiple review types for PR #{pr_number}[/cyan]",
        title="Review Comparison",
        border_style="cyan"
    ))
    
    results = review_manager.compare_reviews(pr_number, review_types)
    
    if results:
        # Display comparison table
        table = Table(title="Review Type Comparison", show_header=True, header_style="bold cyan")
        table.add_column("Review Type", style="cyan", width=15)
        table.add_column("Assessment", style="white", width=15)
        table.add_column("Summary", style="green", min_width=50)
        table.add_column("Confidence", style="yellow", width=10)
        
        for review_type, result in results.items():
            assessment_color = {
                'APPROVE': '[green]APPROVE[/green]',
                'REQUEST_CHANGES': '[red]REQUEST_CHANGES[/red]',
                'COMMENT': '[yellow]COMMENT[/yellow]'
            }.get(result.assessment, result.assessment)
            
            table.add_row(
                review_type.value.title(),
                assessment_color,
                result.summary,
                f"{result.confidence_score:.0%}"
            )
        
        console.print(table)
    else:
        console.print("[yellow]No reviews completed[/yellow]")

def display_review_result(result: ReviewResult) -> None:
    """Display a structured review result"""
    
    # Color-code the assessment
    assessment_color = {
        'APPROVE': '[green]APPROVE ✅[/green]',
        'REQUEST_CHANGES': '[red]REQUEST CHANGES ❌[/red]',
        'COMMENT': '[yellow]COMMENT 💬[/yellow]'
    }.get(result.assessment, result.assessment)
    
    # Create main review panel
    review_content = [
        f"[bold cyan]Review Type:[/bold cyan] {result.review_type.value.title()}",
        f"[bold cyan]Assessment:[/bold cyan] {assessment_color}",
        f"[bold cyan]Confidence:[/bold cyan] {result.confidence_score:.0%}",
        "",
        f"[bold cyan]Summary:[/bold cyan]",
        result.summary,
        "",
        f"[bold cyan]Detailed Feedback:[/bold cyan]",
        result.detailed_feedback
    ]
    
    console.print(Panel(
        '\n'.join(review_content),
        title=f"[cyan]PR #{result.pr_number} Review[/cyan]",
        border_style="cyan",
        padding=(1, 2)
    ))
    
    # Display checklist if available
    if result.checklist_items:
        console.print("\n[bold]📋 Review Checklist[/bold]")
        
        checklist_table = Table(show_header=True, header_style="bold cyan")
        checklist_table.add_column("Status", width=8)
        checklist_table.add_column("Item", min_width=30)
        checklist_table.add_column("Notes", min_width=20)
        
        for item in result.checklist_items:
            status_icon = {
                'passed': '✅ Pass',
                'attention': '⚠️ Review',
                'failed': '❌ Fail',
                'review_needed': '👀 Manual',
                'unknown': '❓ Unknown'
            }.get(item['status'], '❓ Unknown')
            
            checklist_table.add_row(
                status_icon,
                item['item'],
                item['notes']
            )
        
        console.print(checklist_table)
    
    # Display specific issue types if available
    if result.security_issues:
        console.print("\n[bold red]🔒 Security Issues[/bold red]")
        for issue in result.security_issues:
            console.print(f"• {issue}")
    
    if result.dependency_issues:
        console.print("\n[bold yellow]📦 Dependency Issues[/bold yellow]")
        for issue in result.dependency_issues:
            console.print(f"• {issue}")

def show_review_templates() -> None:
    """Show available review templates and their details"""
    
    review_manager = PRReviewManager()
    templates = review_manager.review_templates
    
    console.print(Panel(
        "[cyan]Available Review Templates[/cyan]",
        title="Review Types",
        border_style="cyan"
    ))
    
    for review_type, template in templates.items():
        console.print(f"\n[bold cyan]{template.name}[/bold cyan] ({review_type.value})")
        console.print(f"Focus Areas: {', '.join(template.focus_areas)}")
        console.print(f"Checklist Items: {len(template.checklist)}")
        
        # Show first few checklist items
        if template.checklist:
            console.print("Sample checklist items:")
            for item in template.checklist[:3]:
                console.print(f"  • {item}")
            if len(template.checklist) > 3:
                console.print(f"  ... and {len(template.checklist) - 3} more")

def review_current_branch_enhanced(review_type: str = "quick", auto_post: bool = False) -> None:
    """Enhanced review of current branch with structured review types"""
    
    # Convert string to ReviewType enum
    try:
        review_type_enum = ReviewType(review_type.lower())
    except ValueError:
        available_types = [rt.value for rt in ReviewType]
        console.print(Panel(
            f"[red]Invalid review type: {review_type}[/red]\n\n"
            f"Available types: {', '.join(available_types)}",
            title="Error",
            border_style="red"
        ))
        return
    
    # Check if we're in a PR branch
    github_api = GitHubAPI()
    try:
        current_pr = github_api.get_current_pr()
        
        if current_pr:
            console.print(f"[cyan]Found PR #{current_pr['number']} for current branch[/cyan]")
            enhanced_review_pr(current_pr['number'], review_type, auto_post)
        else:
            # Fall back to original branch review
            review_current_branch(auto_post)
            
    except Exception as e:
        console.print(Panel(
            f"[red]Failed to review branch: {str(e)}[/red]",
            title="Error",
            border_style="red"
        )) 