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

def animated_check_git():
    """Check git repository with animation"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Checking git repository...", total=None)
        result = is_git_repository()
        progress.update(task, completed=True)
        return result

def is_git_repository():
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
    if not animated_check_git():
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
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Get staged changes
        task1 = progress.add_task("Getting staged changes...", total=None)
        diff = get_staged_diff()
        progress.update(task1, completed=True)
        
        # Generate message
        task2 = progress.add_task("Generating commit message...", total=None)
        provider = get_ai_provider()
        model = settings.get_model()
        message = generate_commit_message(diff, provider, model)
        progress.update(task2, completed=True)
    
    # Show the generated message in a panel
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
        create_commit(message)
    else:
        console.print("\n[yellow]Commit cancelled[/yellow]")

if __name__ == "__main__":
    app()
