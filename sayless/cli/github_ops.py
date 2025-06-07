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
        self.repo_url = run_git_command(['config', '--get', 'remote.origin.url']).stdout.strip()
        if "github.com" not in self.repo_url:
            console.print("[red]Not a GitHub repository[/red]")
            sys.exit(1)
        
        # Extract owner and repo from URL
        self.repo_url = self.repo_url.rstrip('.git')
        parts = self.repo_url.split('github.com/')[-1].split('/')
        self.owner = parts[-2]
        self.repo = parts[-1]

    def create_pr(self, title: str, body: str, base: str = "main", labels: List[str] = None) -> Dict:
        """Create a pull request"""
        head = get_current_branch()
        
        url = f"{self.api_url}/repos/{self.owner}/{self.repo}/pulls"
        data = {
            "title": title,
            "body": body,
            "head": head,
            "base": base
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code != 201:
            raise Exception(f"Failed to create PR: {response.json().get('message', '')}")
        
        pr = response.json()
        
        # Add labels if provided
        if labels:
            labels_url = f"{self.api_url}/repos/{self.owner}/{self.repo}/issues/{pr['number']}/labels"
            requests.post(labels_url, headers=self.headers, json=labels)
        
        return pr

    def list_prs(self, state: str = "open") -> List[Dict]:
        """List pull requests"""
        url = f"{self.api_url}/repos/{self.owner}/{self.repo}/pulls"
        params = {"state": state}
        
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to list PRs: {response.json().get('message', '')}")
        
        return response.json()

def generate_pr_content(branch: str = None) -> Dict[str, str]:
    """Generate PR title, body, and labels using AI"""
    if not branch:
        branch = get_current_branch()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Analyzing changes...", total=None)
        
        try:
            # Get base branch
            base_branch = 'main' if run_git_command(['rev-parse', '--verify', 'main'], check=False).returncode == 0 else 'master'
            
            # Get changes
            merge_base = run_git_command(['merge-base', base_branch, branch]).stdout.strip()
            diff = run_git_command(['diff', merge_base, branch]).stdout
            commits = run_git_command(['log', '--oneline', f'{merge_base}..{branch}']).stdout
            
            provider = settings.get_provider()
            model = settings.get_model()
            
            prompt = f"""Based on these changes, generate a pull request title and description.
The changes are:

Commits:
{commits}

Changes:
{diff}

Respond in this format:
<title>
Brief, descriptive PR title
</title>

<body>
## Changes
- Key changes and their purpose

## Testing
- How to test these changes

## Additional Notes
- Any important considerations
</body>

<labels>
Comma-separated list of relevant labels from: feature, bug, documentation, enhancement, refactor, performance, testing, maintenance
</labels>"""
            
            if provider == 'openai':
                api_key = settings.get_openai_api_key()
                ai = OpenAIProvider(api_key)
            else:
                ai = OllamaProvider()
            
            response = ai.generate_commit_message(prompt, model)
            
            # Parse response
            title = response.split('<title>')[1].split('</title>')[0].strip()
            body = response.split('<body>')[1].split('</body>')[0].strip()
            labels = response.split('<labels>')[1].split('</labels>')[0].strip().split(',')
            labels = [l.strip() for l in labels]
            
            progress.update(task, completed=True)
            
            return {
                'title': title,
                'body': body,
                'labels': labels
            }
        except Exception as e:
            progress.update(task, visible=False)
            console.print(Panel(f"[red]Failed to generate PR content: {str(e)}[/red]", title="Error", border_style="red"))
            sys.exit(1)

def create_pr(base: str = None) -> None:
    """Create a pull request with AI-generated content"""
    github = GitHubAPI()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Generate PR content
        task_content = progress.add_task("Generating PR content...", total=None)
        content = generate_pr_content()
        progress.update(task_content, completed=True)
        
        # Show preview
        console.print("\n[bold]Pull Request Preview[/bold]")
        console.print(Panel(
            f"[bold cyan]Title:[/bold cyan] {content['title']}\n\n"
            f"[bold cyan]Body:[/bold cyan]\n{content['body']}\n\n"
            f"[bold cyan]Labels:[/bold cyan] {', '.join(content['labels'])}",
            title="Preview",
            border_style="cyan"
        ))
        
        # Confirm creation
        if not typer.confirm("\n💭 Do you want to create this pull request?"):
            console.print("\n[yellow]PR creation cancelled[/yellow]")
            return
        
        # Create PR
        task_create = progress.add_task("Creating pull request...", total=None)
        try:
            pr = github.create_pr(
                title=content['title'],
                body=content['body'],
                base=base or 'main',
                labels=content['labels']
            )
            progress.update(task_create, completed=True)
            
            console.print(f"\n[bold green]✓[/bold green] Pull request created: [link={pr['html_url']}]{pr['html_url']}[/link]")
        except Exception as e:
            progress.update(task_create, visible=False)
            console.print(Panel(f"[red]Failed to create PR: {str(e)}[/red]", title="Error", border_style="red"))
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
    provider = settings.get_provider()
    model = settings.get_model()
    
    # Get PR details
    files_changed = len(pr['changed_files'])
    additions = pr['additions']
    deletions = pr['deletions']
    
    prompt = f"""Analyze this pull request and provide a one-line insight:

Title: {pr['title']}
Description: {pr['body']}
Files Changed: {files_changed}
Additions: {additions}
Deletions: {deletions}
Labels: {', '.join(l['name'] for l in pr['labels'])}

Focus on risk level, complexity, and key areas of change. Keep it concise."""
    
    try:
        if provider == 'openai':
            api_key = settings.get_openai_api_key()
            ai = OpenAIProvider(api_key)
        else:
            ai = OllamaProvider()
        
        return ai.generate_commit_message(prompt, model).strip()
    except:
        return "Unable to generate insights" 