#!/usr/bin/env python3

import subprocess
import sys
import typer
from rich.console import Console
from rich import print
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from typing import Optional
import os
import time
from config import Config
from ai_providers import OllamaProvider, OpenAIProvider
import datetime
from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta

app = typer.Typer(help="🤖 AI-powered Git commit message generator")
console = Console()
settings = Config()

VALID_PROVIDERS = ["openai", "ollama"]

def show_welcome_message():
    """Show a welcome message with the current configuration"""
    provider = settings.get_provider()
    model = settings.get_model()
    
    table = Table(show_header=False, box=None)
    table.add_row("[bold cyan]Sayless[/bold cyan] 🤖", "AI Commit Message Generator")
    table.add_row("Provider", f"[green]{provider}[/green] {'(default)' if provider == 'openai' else '(local AI)'}")
    table.add_row("Model", f"[green]{model}[/green]")
    
    panel = Panel(table, border_style="cyan")
    console.print(panel)
    console.print()

def ensure_openai_configured():
    """Ensure OpenAI is properly configured or guide the user"""
    if settings.get_provider() != 'openai':
        return
        
    api_key = settings.get_openai_api_key()
    if not api_key:
        panel = Panel(
            "[bold red]OpenAI API Key Required[/bold red]\n\n"
            "[bold]You have two options:[/bold]\n\n"
            "[bold cyan]1. Configure OpenAI (Recommended):[/bold cyan]\n"
            "   Set your API key using one of these methods:\n"
            "   [blue]a) Environment variable:[/blue]\n"
            "      export OPENAI_API_KEY=your_api_key\n"
            "   [blue]b) Configuration command:[/blue]\n"
            "      sayless config --openai-key YOUR_API_KEY\n"
            "   [blue]c) Quick switch command:[/blue]\n"
            "      sayless switch openai --key YOUR_API_KEY\n\n"
            "[bold yellow]2. Switch to Ollama (Local AI):[/bold yellow]\n"
            "   [blue]a) Install Ollama first:[/blue]\n"
            "      Visit https://ollama.ai\n"
            "   [blue]b) Then switch to Ollama:[/blue]\n"
            "      sayless switch ollama",
            title="Configuration Required",
            border_style="red"
        )
        console.print(panel)
        sys.exit(1)

def show_config_status():
    """Show current configuration status in a nice table"""
    table = Table(title="Current Configuration", show_header=False, title_style="bold cyan", border_style="cyan")
    provider = settings.get_provider()
    has_key = bool(settings.get_openai_api_key())
    
    table.add_row("Provider", f"[green]{provider}[/green] {'(default)' if provider == 'openai' else '(local AI)'}")
    table.add_row("Model", f"[green]{settings.get_model()}[/green]")
    if provider == 'openai':
        table.add_row("OpenAI API Key", f"[{'green' if has_key else 'red'}]{'configured' if has_key else 'not configured'}[/{'green' if has_key else 'red'}]")
    
    console.print(table)
    console.print()

def check_git_repo():
    """Check if current directory is a git repository"""
    try:
        subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def has_staged_changes():
    """Check if there are staged changes to commit"""
    try:
        diff = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], 
                                     stderr=subprocess.DEVNULL).decode('utf-8')
        return bool(diff.strip())
    except subprocess.CalledProcessError:
        return False

def get_staged_diff():
    """Get the diff of staged changes"""
    if not check_git_repo():
        panel = Panel(
            "[red]Not in a git repository[/red]\n\n"
            "[yellow]Please run this command inside a git repository.[/yellow]\n"
            "To initialize a new git repository:\n"
            "[blue]  git init[/blue]",
            title="Error",
            border_style="red"
        )
        console.print(panel)
        sys.exit(1)

    if not has_staged_changes():
        panel = Panel(
            "[red]No staged changes found[/red]\n\n"
            "[yellow]Please stage your changes first:[/yellow]\n"
            "[blue]git add <files>[/blue]\n"
            "# or to stage all changes:\n"
            "[blue]git add .[/blue]",
            title="Error",
            border_style="red"
        )
        console.print(panel)
        sys.exit(1)

    try:
        # Get the diff of staged changes
        diff = subprocess.check_output(
            ['git', 'diff', '--cached', '--no-color'],
            stderr=subprocess.PIPE
        ).decode('utf-8')
        
        if not diff.strip():
            console.print(Panel("[red]Staged changes are empty[/red]", title="Error", border_style="red"))
            sys.exit(1)
            
        return diff
    except subprocess.CalledProcessError as e:
        console.print(Panel(f"[red]Failed to get staged changes\nDetails: {e.stderr.decode('utf-8')}[/red]", title="Error", border_style="red"))
        sys.exit(1)

def create_commit(message: str):
    """Create a commit with the given message"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating commit...", total=None)
        try:
            subprocess.run(['git', 'commit', '-m', message], check=True)
            progress.update(task, completed=True)
            console.print("\n[bold green]✓[/bold green] Commit created successfully!")
        except subprocess.CalledProcessError as e:
            progress.update(task, completed=True)
            console.print(Panel(f"[red]Failed to create commit\nDetails: {str(e)}[/red]", title="Error", border_style="red"))
            sys.exit(1)

def get_ai_provider():
    """Get the configured AI provider"""
    provider = settings.get_provider()
    
    if provider == 'openai':
        api_key = settings.get_openai_api_key()
        if not api_key:
            ensure_openai_configured()
        return OpenAIProvider(api_key)
    else:
        return OllamaProvider()

def generate_commit_message(diff: str, provider, model: str) -> str:
    """Generate commit message with fallback handling"""
    try:
        return provider.generate_commit_message(diff, model)
    except Exception as e:
        if isinstance(provider, OpenAIProvider):
            console.print("\n[yellow]OpenAI failed. Trying Ollama as fallback...[/yellow]")
            try:
                with console.status("[bold yellow]Initializing Ollama fallback...[/bold yellow]"):
                    fallback_provider = OllamaProvider()
                return fallback_provider.generate_commit_message(diff, "llama2")
            except Exception:
                console.print(Panel(f"[red]Both OpenAI and Ollama failed.\nOriginal error: {str(e)}[/red]", title="Error", border_style="red"))
                sys.exit(1)
        raise

def get_commit_range(since=None, until=None):
    """Get commit range based on dates"""
    cmd = ['git', 'log', '--no-merges', '--format=%H']
    
    if since:
        cmd.append(f'--since={since}')
    if until:
        cmd.append(f'--until={until}')
    
    try:
        commits = subprocess.check_output(cmd, stderr=subprocess.PIPE).decode('utf-8').strip().split('\n')
        return commits if commits and commits[0] else []
    except subprocess.CalledProcessError:
        return []

def get_commit_details(commit_hash):
    """Get details for a specific commit"""
    try:
        details = subprocess.check_output(
            ['git', 'show', '--no-color', '--format=%s%n%b', commit_hash],
            stderr=subprocess.PIPE
        ).decode('utf-8').strip()
        return details
    except subprocess.CalledProcessError:
        return None

def generate_summary(commits, provider, model):
    """Generate a summary of changes from commits"""
    if not commits:
        return "No changes found in the specified period."
    
    # Get all commit messages and diffs
    commit_details = []
    for commit in commits[:10]:  # Limit to last 10 commits for reasonable context
        details = get_commit_details(commit)
        if details:
            commit_details.append(details)
    
    commit_text = "\n\n".join(commit_details)
    
    prompt = f"""Based on these recent commits, generate a concise summary of changes.
Group related changes together and highlight major updates.
Focus on the business impact and key features/fixes.

Recent commits:

{commit_text}

Generate a summary in this format:
## Major Changes
- (grouped major changes)

## Features
- (new features)

## Fixes & Improvements
- (bug fixes and improvements)

Keep each bullet point concise and clear."""

    try:
        return provider.generate_commit_message(prompt, model)
    except Exception as e:
        if isinstance(provider, OpenAIProvider):
            console.print("[yellow]OpenAI failed, trying Ollama fallback...[/yellow]")
            try:
                fallback_provider = OllamaProvider()
                return fallback_provider.generate_commit_message(prompt, "llama2")
            except Exception:
                raise Exception(f"Both providers failed. Original error: {str(e)}")
        raise

def parse_time_interval(interval):
    """Parse time interval string into timedelta"""
    if not interval:
        return relativedelta(days=1)  # Default to last 24 hours
        
    units = {
        'd': 'days',
        'w': 'weeks',
        'm': 'months',
        'y': 'years',
        'h': 'hours'
    }
    
    unit = interval[-1].lower()
    try:
        value = int(interval[:-1])
        if unit in units:
            return relativedelta(**{units[unit]: value})
    except ValueError:
        pass
    
    # If parsing fails, try to parse as a date
    try:
        since_date = parse_date(interval)
        return since_date
    except ValueError:
        raise ValueError(
            "Invalid time interval. Use format: <number><unit> or a date.\n"
            "Units: h (hours), d (days), w (weeks), m (months), y (years)\n"
            "Example: 2w (2 weeks), 3m (3 months), 2023-01-01"
        )

def format_date(date_str):
    """Convert date string to human readable format"""
    try:
        if isinstance(date_str, str):
            date = parse_date(date_str)
        else:
            date = date_str
        return date.strftime("%A, %B %d, %Y")  # e.g., "Tuesday, May 15, 2023"
    except:
        return date_str

@app.command()
def switch(
    provider: str = typer.Argument(..., help="AI provider to use (openai or ollama)"),
    key: Optional[str] = typer.Option(None, "--key", help="OpenAI API key (required for OpenAI)"),
    model: Optional[str] = typer.Option(None, "--model", help="Model to use (optional)"),
):
    """Quickly switch between AI providers (OpenAI/Ollama)"""
    
    # Validate provider
    provider = provider.lower()
    if provider not in VALID_PROVIDERS:
        console.print(Panel(
            f"[red]Invalid provider '{provider}'[/red]\n"
            f"[yellow]Valid providers are: {', '.join(VALID_PROVIDERS)}[/yellow]",
            title="Error",
            border_style="red"
        ))
        sys.exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Switching to {provider}...", total=None)
        
        if provider == "openai":
            # Check for API key
            api_key = key or os.getenv("OPENAI_API_KEY") or settings.get_openai_api_key()
            if not api_key:
                progress.update(task, completed=True)
                console.print(Panel(
                    "[red]OpenAI API key required[/red]\n"
                    "[yellow]Provide it using one of these methods:[/yellow]\n"
                    "[blue]1. With the switch command:[/blue]\n"
                    "   sayless switch openai --key YOUR_API_KEY\n"
                    "[blue]2. Environment variable:[/blue]\n"
                    "   export OPENAI_API_KEY=your_api_key",
                    title="Error",
                    border_style="red"
                ))
                sys.exit(1)
            
            # Configure OpenAI
            settings.set_openai_api_key(api_key)
            settings.set_provider('openai')
            if not model:
                settings.set_model('gpt-4')  # Set default OpenAI model
            
        elif provider == "ollama":
            settings.set_provider('ollama')
            if not model:
                settings.set_model('llama2')  # Set default Ollama model
        
        if model:
            settings.set_model(model)
        
        time.sleep(0.5)  # Add a small delay for better UX
        progress.update(task, completed=True)
    
    # Show success message and new configuration
    console.print(f"\n[bold green]✓[/bold green] Successfully switched to {provider}!")
    show_config_status()

@app.command()
def config(
    openai_key: Optional[str] = typer.Option(None, "--openai-key", help="Set OpenAI API key"),
    use_openai: bool = typer.Option(False, "--use-openai", help="Use OpenAI (default)"),
    use_ollama: bool = typer.Option(False, "--use-ollama", help="Use Ollama (local AI)"),
    model: Optional[str] = typer.Option(None, "--model", help="Set the model to use"),
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
):
    """Configure the AI provider and settings"""
    if show:
        show_config_status()
        provider = settings.get_provider()
        has_key = bool(settings.get_openai_api_key())
        
        if not has_key and provider == 'openai':
            console.print(Panel(
                "[yellow]OpenAI API key not configured[/yellow]\n"
                "[blue]Quick setup:[/blue]\n"
                "  sayless switch openai --key YOUR_API_KEY",
                title="Warning",
                border_style="yellow"
            ))
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Updating configuration...", total=None)

        if openai_key:
            settings.set_openai_api_key(openai_key)
            settings.set_provider('openai')  # Automatically switch to OpenAI when key is provided

        if use_openai and use_ollama:
            progress.update(task, completed=True)
            console.print(Panel("[red]Cannot use both OpenAI and Ollama at the same time[/red]", title="Error", border_style="red"))
            sys.exit(1)

        if use_openai:
            settings.set_provider('openai')
            if not settings.get_openai_api_key():
                progress.update(task, completed=True)
                console.print(Panel(
                    "[yellow]OpenAI API key not configured[/yellow]\n"
                    "[blue]Quick setup:[/blue]\n"
                    "  sayless switch openai --key YOUR_API_KEY",
                    title="Warning",
                    border_style="yellow"
                ))
        elif use_ollama:
            settings.set_provider('ollama')

        if model:
            settings.set_model(model)

        time.sleep(0.5)  # Add a small delay for better UX
        progress.update(task, completed=True)

    console.print("\n[bold green]✓[/bold green] Configuration updated successfully!")
    show_config_status()

@app.command()
def generate(
    preview: bool = typer.Option(False, help="Preview the commit message without creating the commit"),
):
    """Generate a commit message for staged changes and create the commit"""
    
    show_welcome_message()
    
    # Ensure OpenAI is properly configured if it's the selected provider
    ensure_openai_configured()
    
    # Create a single progress instance for all tasks
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    )
    
    message = None  # Initialize message variable
    
    with progress:
        # Check git repository
        task_check = progress.add_task("Checking git repository...", total=None)
        if not check_git_repo():
            progress.update(task_check, visible=False)
            panel = Panel(
                "[red]Not in a git repository[/red]\n\n"
                "[yellow]Please run this command inside a git repository.[/yellow]\n"
                "To initialize a new git repository:\n"
                "[blue]  git init[/blue]",
                title="Error",
                border_style="red"
            )
            console.print(panel)
            sys.exit(1)
        progress.update(task_check, completed=True)
        
        # Get staged changes
        task_diff = progress.add_task("Getting staged changes...", total=None)
        diff = get_staged_diff()
        progress.update(task_diff, completed=True)
        
        # Generate message
        task_gen = progress.add_task("Generating commit message...", total=None)
        provider = get_ai_provider()
        model = settings.get_model()
        
        try:
            message = provider.generate_commit_message(diff, model)
        except Exception as e:
            if isinstance(provider, OpenAIProvider):
                progress.update(task_gen, description="OpenAI failed, trying Ollama fallback...")
                try:
                    fallback_provider = OllamaProvider()
                    message = fallback_provider.generate_commit_message(diff, "llama2")
                except Exception:
                    progress.update(task_gen, visible=False)
                    console.print(Panel(
                        f"[red]Both OpenAI and Ollama failed.\nOriginal error: {str(e)}[/red]",
                        title="Error",
                        border_style="red"
                    ))
                    sys.exit(1)
            else:
                raise
        
        progress.update(task_gen, completed=True)

    # Show the generated message in a panel
    if message:
        console.print(Panel(
            f"[yellow]{message}[/yellow]",
            title="Generated Commit Message",
            border_style="green"
        ))

        if preview:
            console.print("\n[blue]Preview mode: Commit not created[/blue]")
            return

        # Ask for confirmation with styled prompt
        if typer.confirm("\n💭 Do you want to create the commit with this message?"):
            with progress:
                task_commit = progress.add_task("Creating commit...", total=None)
                try:
                    subprocess.run(['git', 'commit', '-m', message], check=True)
                    progress.update(task_commit, completed=True)
                    console.print("\n[bold green]✓[/bold green] Commit created successfully!")
                except subprocess.CalledProcessError as e:
                    progress.update(task_commit, visible=False)
                    console.print(Panel(
                        f"[red]Failed to create commit\nDetails: {str(e)}[/red]",
                        title="Error",
                        border_style="red"
                    ))
                    sys.exit(1)
        else:
            console.print("\n[yellow]Commit cancelled[/yellow]")

@app.command()
def summary(
    preview: bool = typer.Option(False, help="Preview the summary without saving"),
    save: bool = typer.Option(False, help="Save the summary to a file"),
):
    """Generate a summary of recent changes (last 24 hours)"""
    show_welcome_message()
    ensure_openai_configured()
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    )
    
    with progress:
        # Get recent commits
        task_commits = progress.add_task("Getting recent commits...", total=None)
        commits = get_commit_range(since="1 day ago")
        progress.update(task_commits, completed=True)
        
        if not commits:
            console.print(Panel(
                "[yellow]No changes found in the last 24 hours.[/yellow]",
                title="Summary",
                border_style="yellow"
            ))
            return
        
        # Generate summary
        task_summary = progress.add_task("Generating summary...", total=None)
        provider = get_ai_provider()
        model = settings.get_model()
        
        try:
            summary_text = generate_summary(commits, provider, model)
            progress.update(task_summary, completed=True)
        except Exception as e:
            progress.update(task_summary, visible=False)
            console.print(Panel(
                f"[red]Failed to generate summary\nError: {str(e)}[/red]",
                title="Error",
                border_style="red"
            ))
            return
    
    # Get yesterday's date in human readable format
    yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
    date_str = format_date(yesterday)
    
    # Show the summary
    console.print(Panel(
        f"[green]{summary_text}[/green]",
        title=f"Changes Summary (Since {date_str})",
        border_style="cyan"
    ))
    
    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_{timestamp}.md"
        try:
            with open(filename, 'w') as f:
                f.write(f"# Changes Summary (Since {date_str})\n\n")
                f.write(summary_text)
            console.print(f"\n[green]Summary saved to: {filename}[/green]")
        except Exception as e:
            console.print(f"\n[red]Failed to save summary: {str(e)}[/red]")

@app.command()
def since(
    interval: str = typer.Argument(..., help="Time interval (e.g., 2w, 3m, 2023-01-01)"),
    until: str = typer.Option(None, help="End date (default: now)"),
    preview: bool = typer.Option(False, help="Preview the summary without saving"),
    save: bool = typer.Option(False, help="Save the summary to a file"),
):
    """Generate a summary of changes since a specific time or date"""
    show_welcome_message()
    ensure_openai_configured()
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    )
    
    try:
        since_time = parse_time_interval(interval)
        if isinstance(since_time, datetime.datetime):
            since_str = since_time.strftime("%Y-%m-%d")
            since_display = format_date(since_time)
        else:
            since_date = datetime.datetime.now() - since_time
            since_str = since_date.strftime("%Y-%m-%d")
            since_display = format_date(since_date)
        
        until_display = format_date(until) if until else "now"
        
    except ValueError as e:
        console.print(Panel(
            f"[red]{str(e)}[/red]",
            title="Error",
            border_style="red"
        ))
        return
    
    with progress:
        # Get commits for the period
        task_commits = progress.add_task("Getting commits...", total=None)
        commits = get_commit_range(since=since_str, until=until)
        progress.update(task_commits, completed=True)
        
        if not commits:
            console.print(Panel(
                f"[yellow]No changes found between {since_display} and {until_display}.[/yellow]",
                title="Summary",
                border_style="yellow"
            ))
            return
        
        # Generate summary
        task_summary = progress.add_task("Generating summary...", total=None)
        provider = get_ai_provider()
        model = settings.get_model()
        
        try:
            summary_text = generate_summary(commits, provider, model)
            progress.update(task_summary, completed=True)
        except Exception as e:
            progress.update(task_summary, visible=False)
            console.print(Panel(
                f"[red]Failed to generate summary\nError: {str(e)}[/red]",
                title="Error",
                border_style="red"
            ))
            return
    
    # Show the summary
    title = f"Changes Summary (From {since_display} to {until_display})"
    console.print(Panel(
        f"[green]{summary_text}[/green]",
        title=title,
        border_style="cyan"
    ))
    
    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_since_{interval.replace('/', '_')}_{timestamp}.md"
        try:
            with open(filename, 'w') as f:
                f.write(f"# {title}\n\n")
                f.write(summary_text)
            console.print(f"\n[green]Summary saved to: {filename}[/green]")
        except Exception as e:
            console.print(f"\n[red]Failed to save summary: {str(e)}[/red]")

if __name__ == "__main__":
    app()
