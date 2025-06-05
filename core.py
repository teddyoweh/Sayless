#!/usr/bin/env python3

import subprocess
import sys
import json
import requests
import typer
from rich.console import Console
from rich import print
from typing import Optional
from setup import ensure_ollama_ready

app = typer.Typer()
console = Console()

def get_last_commit_diff():
    """Get the diff of the last commit"""
    try:
        # Get the diff of the last commit
        diff = subprocess.check_output(
            ['git', 'show', '--pretty=format:"commit %H%n%s%n%b"', '--no-color'],
            stderr=subprocess.PIPE
        ).decode('utf-8')
        return diff
    except subprocess.CalledProcessError as e:
        console.print("[red]Error: Not a git repository or no commits exist[/red]")
        sys.exit(1)

def generate_commit_message(diff: str, model: str = "llama2") -> str:
    """Generate a commit message using Ollama"""
    prompt = f"""Based on the following git diff, generate a clear and concise commit message that follows conventional commits format.
The message should be in the format: <type>(<scope>): <description>

Types can be:
- feat: A new feature
- fix: A bug fix
- docs: Documentation only changes
- style: Changes that do not affect the meaning of the code
- refactor: A code change that neither fixes a bug nor adds a feature
- perf: A code change that improves performance
- test: Adding missing tests or correcting existing tests
- chore: Changes to the build process or auxiliary tools

Here's the diff:

{diff}

Generate only the commit message without any explanation."""

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False
            }
        )
        response.raise_for_status()
        result = response.json()
        return result['response'].strip()
    except requests.exceptions.RequestException as e:
        console.print("[red]Error: Failed to connect to Ollama. Make sure it's running.[/red]")
        console.print(f"[red]Details: {str(e)}[/red]")
        sys.exit(1)

@app.command()
def generate(
    model: str = typer.Option("llama2", help="The Ollama model to use (default: llama2)"),
):
    """Generate a commit message for the last commit using AI"""
    
    # Ensure Ollama is ready with the specified model
    ensure_ollama_ready(model)
    
    with console.status("[bold green]Getting last commit diff..."):
        diff = get_last_commit_diff()
    
    with console.status("[bold green]Generating commit message..."):
        message = generate_commit_message(diff, model)
    
    print("\n[bold green]Generated commit message:[/bold green]")
    print(f"[yellow]{message}[/yellow]\n")

if __name__ == "__main__":
    app()
