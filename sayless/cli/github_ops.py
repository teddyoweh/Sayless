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
from .git_ops import run_git_command, get_current_branch

console = Console()
settings = Config()

class GitHubAPI:
    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            console.print(Panel(
                "[red]GitHub token not found[/red]\n\n"
                "[yellow]Please set your GitHub token:[/yellow]\n"
                "1. Create a token at https://github.com/settings/tokens\n"
                "2. Set it in your environment:\n"
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
            # Validate parameters first
            self.validate_pr_params(head, base)
            
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
        base_branch = 'main' if run_git_command(['rev-parse', '--verify', 'main'], check=False).returncode == 0 else 'master'
        
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
        
        provider = settings.get_provider()
        model = settings.get_model()
        
        prompt = f"""Based on these changes, generate a pull request title and description.
The changes are:

Commits:
{commits}

Changes:
{diff}

Generate a clear, descriptive PR that explains the changes, their purpose, and how to test them.
Focus on making the description helpful for reviewers.

DO NOT include any XML-like tags in your response.
Instead, format your response exactly like this:

Title: Brief, descriptive title (use conventional commit format)

## Changes
- Key changes and their purpose (bullet points)
- Impact of these changes
- Any breaking changes or dependencies

## Testing
- How to test these changes
- Expected outcomes
- Any specific test cases

## Additional Notes
- Important considerations
- Related issues or PRs
- Any follow-up tasks needed

Labels: comma-separated list from [feature, bug, documentation, enhancement, refactor, performance, testing, maintenance]"""
        
        if provider == 'openai':
            api_key = settings.get_openai_api_key()
            ai = OpenAIProvider(api_key)
        else:
            ai = OllamaProvider()
        
        response = ai.generate_commit_message(prompt, model)
        
        # Parse response
        try:
            # Extract title (first non-empty line)
            lines = [l for l in response.split('\n') if l.strip()]
            title = lines[0].replace('Title:', '').strip()
            
            # Extract body (everything after title)
            body = '\n'.join(lines[1:]).strip()
            
            # Extract labels (last line if it starts with "Labels:")
            labels = []
            if lines[-1].lower().startswith('labels:'):
                labels_text = lines[-1].split(':', 1)[1].strip()
                labels = [l.strip() for l in labels_text.split(',') if l.strip()]
                # Remove labels line from body
                body = '\n'.join(lines[1:-1]).strip()
            
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
            raise ValueError(f"Failed to parse AI response: {str(e)}\nResponse:\n{response}")
            
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
            try:
                github.validate_pr_params(head, base or 'main')
            except ValueError as e:
                if auto_push and "not found on GitHub" in str(e):
                    # Try to push the branch
                    try:
                        if push_branch(head):
                            # Retry validation after successful push
                            github.validate_pr_params(head, base or 'main')
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
                base=base or 'main',
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
            
            # Show details if requested
            if show_details:
                task_insights = progress.add_task("Generating PR insights...", total=None)
                try:
                    insights = get_pr_insights(pr)
                    progress.update(task_insights, completed=True)
                    console.print(Panel(
                        f"[bold cyan]AI Insights:[/bold cyan]\n{insights}",
                        title="PR Analysis",
                        border_style="cyan",
                        padding=(1, 2)
                    ))
                except Exception:
                    progress.update(task_insights, visible=False)
                    console.print("[yellow]Could not generate PR insights[/yellow]")
            
        except Exception as e:
            progress.update(task_create, visible=False)
            console.print(Panel(
                f"[red]Failed to create PR: {str(e)}[/red]",
                title="Error",
                border_style="red"
            ))
            sys.exit(1)

def list_prs(show_details: bool = False) -> None:
    """List pull requests with optional AI-generated insights"""
    github = GitHubAPI()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Getting pull requests...", total=None)
        
        try:
            prs = github.list_prs()
            progress.update(task, completed=True)
            
            if not prs:
                console.print("[yellow]No open pull requests found[/yellow]")
                return
            
            # Create table
            table = Table(title="Pull Requests", show_header=True, header_style="bold cyan")
            table.add_column("#", style="cyan", justify="right")
            table.add_column("Title", style="white")
            table.add_column("Author", style="yellow")
            table.add_column("Status", style="green")
            if show_details:
                table.add_column("AI Insights", style="magenta")
            
            for pr in prs:
                status = []
                if pr['draft']:
                    status.append("[yellow]Draft[/yellow]")
                if pr['mergeable']:
                    status.append("[green]Ready[/green]")
                else:
                    status.append("[red]Conflicts[/red]")
                
                row = [
                    str(pr['number']),
                    pr['title'],
                    pr['user']['login'],
                    " ".join(status)
                ]
                
                if show_details:
                    # Get AI insights
                    task_insights = progress.add_task(f"Analyzing PR #{pr['number']}...", total=None)
                    try:
                        insights = get_pr_insights(pr)
                        progress.update(task_insights, completed=True)
                        row.append(insights)
                    except Exception:
                        progress.update(task_insights, visible=False)
                        row.append("[red]Failed to generate insights[/red]")
                
                table.add_row(*row)
            
            console.print(table)
            
        except Exception as e:
            progress.update(task, visible=False)
            console.print(Panel(f"[red]Failed to list PRs: {str(e)}[/red]", title="Error", border_style="red"))
            sys.exit(1)

def get_pr_insights(pr: Dict) -> str:
    """Generate AI insights for a pull request"""
    try:
        provider = settings.get_provider()
        model = settings.get_model()
        
        # Get PR details with error handling
        try:
            files_changed = pr.get('changed_files', 0)
            additions = pr.get('additions', 0)
            deletions = pr.get('deletions', 0)
            labels = [l.get('name', '') for l in pr.get('labels', [])]
            
            # Get the diff for better context
            head = pr.get('head', {}).get('sha')
            base = pr.get('base', {}).get('sha')
            if head and base:
                diff = run_git_command(['diff', '--stat', base, head]).stdout
            else:
                diff = "Diff not available"
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not get complete PR details: {str(e)}[/yellow]")
            diff = "Diff not available"
        
        prompt = f"""Analyze this pull request and provide a concise but informative insight:

Title: {pr.get('title', 'No title')}
Description: {pr.get('body', 'No description')}
Files Changed: {files_changed}
Additions: {additions}
Deletions: {deletions}
Labels: {', '.join(labels)}

Changes:
{diff}

Provide a one-line analysis that covers:
1. The main purpose of the changes
2. Complexity level (low/medium/high)
3. Potential impact
4. Any notable concerns

Keep it concise but informative."""
        
        try:
            if provider == 'openai':
                api_key = settings.get_openai_api_key()
                ai = OpenAIProvider(api_key)
            else:
                ai = OllamaProvider()
            
            insight = ai.generate_commit_message(prompt, model).strip()
            
            # Validate the insight
            if not insight or insight.lower().startswith(('error', 'i apologize', 'i\'m sorry')):
                return "Simple change with low complexity. Review recommended for validation."
                
            return insight
            
        except Exception as e:
            console.print(f"[yellow]Warning: AI generation failed: {str(e)}[/yellow]")
            # Provide a basic insight based on available data
            complexity = "high" if files_changed > 10 or (additions + deletions) > 500 else "medium" if files_changed > 5 or (additions + deletions) > 200 else "low"
            return f"Changes affecting {files_changed} files with {complexity} complexity. Manual review recommended."
            
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to generate insights: {str(e)}[/yellow]")
        return "Unable to analyze PR. Please review changes manually." 