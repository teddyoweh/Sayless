#!/usr/bin/env python3

import subprocess
import sys
import typer
from rich.console import Console
from rich import print
from typing import Optional
import os
from config import Config
from ai_providers import OllamaProvider, OpenAIProvider

app = typer.Typer()
console = Console()
settings = Config()

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
    if not is_git_repository():
        console.print("[red]Error: Not in a git repository[/red]")
        console.print("[yellow]Please run this command inside a git repository.[/yellow]")
        console.print("[yellow]To initialize a new git repository:[/yellow]")
        console.print("[blue]  git init[/blue]")
        sys.exit(1)

    if not has_staged_changes():
        console.print("[red]Error: No staged changes found[/red]")
        console.print("[yellow]Please stage your changes first:[/yellow]")
        console.print("[blue]  git add <files>[/blue]")
        console.print("[blue]  # or to stage all changes:[/blue]")
        console.print("[blue]  git add .[/blue]")
        sys.exit(1)

    try:
        # Get the diff of staged changes
        diff = subprocess.check_output(
            ['git', 'diff', '--cached', '--no-color'],
            stderr=subprocess.PIPE
        ).decode('utf-8')
        
        if not diff.strip():
            console.print("[red]Error: Staged changes are empty[/red]")
            sys.exit(1)
            
        return diff
    except subprocess.CalledProcessError as e:
        console.print("[red]Error: Failed to get staged changes[/red]")
        console.print(f"[red]Details: {e.stderr.decode('utf-8')}[/red]")
        sys.exit(1)

def create_commit(message: str):
    """Create a commit with the given message"""
    try:
        subprocess.run(['git', 'commit', '-m', message], check=True)
        console.print("[green]Successfully created commit![/green]")
    except subprocess.CalledProcessError as e:
        console.print("[red]Error: Failed to create commit[/red]")
        console.print(f"[red]Details: {str(e)}[/red]")
        sys.exit(1)

def get_ai_provider():
    """Get the configured AI provider"""
    provider = settings.get_provider()
    
    if provider == 'openai':
        api_key = settings.get_openai_api_key()
        if not api_key:
            console.print("[red]Error: OpenAI API key not found[/red]")
            console.print("[yellow]Please set your OpenAI API key using one of these methods:[/yellow]")
            console.print("[blue]1. Environment variable:[/blue]")
            console.print("   export OPENAI_API_KEY=your_api_key")
            console.print("[blue]2. Configuration command:[/blue]")
            console.print("   sayless config --openai-key YOUR_API_KEY")
            console.print("\n[yellow]Or switch to Ollama (local AI) with:[/yellow]")
            console.print("   sayless config --use-ollama")
            sys.exit(1)
        return OpenAIProvider(api_key)
    else:
        return OllamaProvider()

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
        console.print("\n[bold]Current Configuration:[/bold]")
        console.print(f"Provider: [green]{settings.get_provider()}[/green]")
        console.print(f"Model: [green]{settings.get_model()}[/green]")
        console.print(f"OpenAI API Key: [green]{'configured' if settings.get_openai_api_key() else 'not configured'}[/green]")
        return

    if openai_key:
        settings.set_openai_api_key(openai_key)
        console.print("[green]OpenAI API key configured successfully![/green]")

    if use_openai and use_ollama:
        console.print("[red]Error: Cannot use both OpenAI and Ollama at the same time[/red]")
        sys.exit(1)

    if use_openai:
        settings.set_provider('openai')
        if not settings.get_openai_api_key():
            console.print("[yellow]Warning: OpenAI API key not configured[/yellow]")
            console.print("Set it with: [blue]sayless config --openai-key YOUR_API_KEY[/blue]")
    elif use_ollama:
        settings.set_provider('ollama')
        console.print("[yellow]Note: Switched to Ollama (local AI)[/yellow]")

    if model:
        settings.set_model(model)
        console.print(f"[green]Model set to: {model}[/green]")

@app.command()
def generate(
    preview: bool = typer.Option(False, help="Preview the commit message without creating the commit"),
):
    """Generate a commit message for staged changes and create the commit"""
    
    with console.status("[bold green]Getting staged changes..."):
        diff = get_staged_diff()
    
    provider = get_ai_provider()
    model = settings.get_model()
    
    with console.status("[bold green]Generating commit message..."):
        try:
            message = provider.generate_commit_message(diff, model)
        except Exception as e:
            if settings.get_provider() == 'openai':
                console.print("[yellow]OpenAI failed. Trying Ollama as fallback...[/yellow]")
                try:
                    provider = OllamaProvider()
                    message = provider.generate_commit_message(diff, "llama2")
                except Exception:
                    console.print("[red]Both OpenAI and Ollama failed.[/red]")
                    console.print(f"[red]Original error: {str(e)}[/red]")
                    sys.exit(1)
    
    print("\n[bold green]Generated commit message:[/bold green]")
    print(f"[yellow]{message}[/yellow]\n")

    if preview:
        console.print("[blue]Preview mode: Commit not created[/blue]")
        return

    if typer.confirm("Do you want to create the commit with this message?"):
        create_commit(message)
    else:
        console.print("[yellow]Commit cancelled[/yellow]")

if __name__ == "__main__":
    app()
