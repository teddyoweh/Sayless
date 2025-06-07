import subprocess
import sys
from typing import List, Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import re
from .ai_providers import OpenAIProvider, OllamaProvider
from .config import Config

console = Console()
settings = Config()

def run_git_command(command: List[str], check=True, capture_output=True) -> subprocess.CompletedProcess:
    """Run a git command and handle errors"""
    try:
        return subprocess.run(['git'] + command, check=check, capture_output=capture_output, text=True)
    except subprocess.CalledProcessError as e:
        console.print(Panel(
            f"[red]Git command failed: {' '.join(['git'] + command)}\nError: {e.stderr}[/red]",
            title="Error",
            border_style="red"
        ))
        sys.exit(1)

def get_current_branch() -> str:
    """Get the name of the current branch"""
    result = run_git_command(['branch', '--show-current'])
    return result.stdout.strip()

def sanitize_branch_name(name: str) -> str:
    """Convert a string into a valid git branch name"""
    # Convert to lowercase
    name = name.lower()
    # Replace spaces and special characters with hyphens
    name = re.sub(r'[^a-z0-9]+', '-', name)
    # Remove leading and trailing hyphens
    name = name.strip('-')
    return name

def get_branch_type(description: str) -> str:
    """Use AI to determine the branch type based on description"""
    provider = settings.get_provider()
    model = settings.get_model()
    
    prompt = f"""Based on this feature description, determine the most appropriate branch type prefix.
Choose one of: feat, fix, docs, style, refactor, perf, test, chore

Description: {description}

Respond with ONLY the type (e.g., 'feat' or 'fix')."""

    try:
        if provider == 'openai':
            api_key = settings.get_openai_api_key()
            ai = OpenAIProvider(api_key)
        else:
            ai = OllamaProvider()
        
        branch_type = ai.generate_commit_message(prompt, model).strip().lower()
        
        # Validate the response
        valid_types = {'feat', 'fix', 'docs', 'style', 'refactor', 'perf', 'test', 'chore'}
        if branch_type not in valid_types:
            return 'feat'  # Default to feat if AI response is invalid
        
        return branch_type
    except:
        return 'feat'  # Default to feat if AI fails

def get_staged_changes_description() -> str:
    """Get a description of staged changes using AI"""
    try:
        # Get staged diff
        diff = run_git_command(['diff', '--cached', '--no-color']).stdout
        if not diff.strip():
            raise ValueError("No staged changes found. Stage your changes first with 'git add'")

        provider = settings.get_provider()
        model = settings.get_model()
        
        prompt = """Based on these staged changes, provide a brief, descriptive title for a branch name.
Keep it concise and focused on the main purpose of the changes.
Respond with ONLY the title, no additional text or formatting.

Changes:
{diff}"""
        
        if provider == 'openai':
            api_key = settings.get_openai_api_key()
            ai = OpenAIProvider(api_key)
        else:
            ai = OllamaProvider()
        
        description = ai.generate_commit_message(prompt, model).strip()
        return description
    except Exception as e:
        raise ValueError(f"Failed to generate branch name: {str(e)}")

def create_branch(description: str = None, checkout: bool = True, generate: bool = False) -> str:
    """Create a new branch with an AI-generated name based on description or staged changes"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        try:
            if generate:
                # Generate description from staged changes
                task_desc = progress.add_task("Analyzing staged changes...", total=None)
                description = get_staged_changes_description()
                progress.update(task_desc, completed=True)
            elif not description:
                raise ValueError("Either provide a description or use --generate (-g)")
            
            # Determine branch type
            task_type = progress.add_task("Determining branch type...", total=None)
            branch_type = get_branch_type(description)
            progress.update(task_type, completed=True)
            
            # Generate branch name
            task_name = progress.add_task("Generating branch name...", total=None)
            branch_name = f"{branch_type}/{sanitize_branch_name(description)}"
            progress.update(task_name, completed=True)
            
            # Create and checkout branch
            task_create = progress.add_task(f"Creating branch {branch_name}...", total=None)
            if checkout:
                run_git_command(['checkout', '-b', branch_name])
            else:
                run_git_command(['branch', branch_name])
            progress.update(task_create, completed=True)
            
            console.print(f"\n[bold green]✓[/bold green] Created branch: [cyan]{branch_name}[/cyan]")
            if checkout:
                console.print(f"[green]Switched to branch: [cyan]{branch_name}[/cyan][/green]")
            
            return branch_name
        except Exception as e:
            console.print(Panel(
                f"[red]{str(e)}[/red]\n\n"
                "[yellow]To create a branch, either:[/yellow]\n"
                "1. Provide a description:\n"
                "   [blue]sl branch \"add user authentication\"[/blue]\n"
                "2. Stage changes and use auto-generate:\n"
                "   [blue]git add .[/blue]\n"
                "   [blue]sl branch -g[/blue]",
                title="Error",
                border_style="red"
            ))
            sys.exit(1)

def list_branches(show_details: bool = False) -> List[Dict[str, str]]:
    """List all branches with optional AI-generated summaries"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Getting branches...", total=None)
        
        # Get all branches
        result = run_git_command(['branch', '--format=%(refname:short)'])
        branches = [b.strip() for b in result.stdout.splitlines()]
        progress.update(task, completed=True)
        
        if not branches:
            console.print("[yellow]No branches found[/yellow]")
            return []
        
        # Create table
        table = Table(title="Git Branches", show_header=True, header_style="bold cyan")
        table.add_column("Branch", style="cyan")
        table.add_column("Last Commit", style="yellow")
        if show_details:
            table.add_column("Summary", style="green")
        
        branch_info = []
        for branch in branches:
            # Get last commit info
            commit_info = run_git_command(['log', '-1', '--format=%h %s', branch])
            last_commit = commit_info.stdout.strip()
            
            info = {
                'name': branch,
                'last_commit': last_commit,
            }
            
            if show_details:
                # Get branch summary using AI
                task_summary = progress.add_task(f"Analyzing branch {branch}...", total=None)
                try:
                    summary = get_branch_summary(branch)
                    progress.update(task_summary, completed=True)
                    info['summary'] = summary
                except Exception as e:
                    progress.update(task_summary, visible=False)
                    info['summary'] = "[red]Failed to generate summary[/red]"
            
            branch_info.append(info)
            
            # Add to table
            if show_details:
                table.add_row(branch, last_commit, info.get('summary', ''))
            else:
                table.add_row(branch, last_commit)
        
        console.print(table)
        return branch_info

def get_branch_summary(branch: str) -> str:
    """Generate an AI summary of the branch's changes"""
    # Get branch changes
    try:
        # Find the merge base with main/master
        base_branch = 'main' if run_git_command(['rev-parse', '--verify', 'main'], check=False).returncode == 0 else 'master'
        merge_base = run_git_command(['merge-base', base_branch, branch]).stdout.strip()
        
        # Get diff summary
        diff = run_git_command(['diff', '--stat', merge_base, branch]).stdout
        commits = run_git_command(['log', '--oneline', f'{merge_base}..{branch}']).stdout
        
        provider = settings.get_provider()
        model = settings.get_model()
        
        prompt = f"""Provide a one-line summary of these branch changes:

Commits:
{commits}

Changes:
{diff}

Keep the summary concise and focused on the main purpose of the changes."""

        if provider == 'openai':
            api_key = settings.get_openai_api_key()
            ai = OpenAIProvider(api_key)
        else:
            ai = OllamaProvider()
        
        return ai.generate_commit_message(prompt, model).strip()
    except:
        return "No changes or unable to generate summary" 