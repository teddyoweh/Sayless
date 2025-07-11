from abc import ABC, abstractmethod
import requests
from openai import OpenAI
from rich.console import Console
from .ollama_setup import ensure_ollama_ready

console = Console()

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class AIProvider(ABC):
    @abstractmethod
    def generate_commit_message(self, diff: str, model: str) -> str:
        pass
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough estimation of tokens (4 characters per token average)"""
        return len(text) // 4
    
    @staticmethod
    def truncate_diff_intelligently(diff: str, max_tokens: int = 60000) -> str:
        """Intelligently truncate diff to fit within token limits"""
        max_chars = max_tokens * 4  # Rough character estimate
        
        if len(diff) <= max_chars:
            return diff
            
        lines = diff.split('\n')
        truncated_lines = []
        current_length = 0
        
        # Prioritize added/removed lines over context
        for line in lines:
            if current_length + len(line) + 1 > max_chars:
                truncated_lines.append("... [DIFF TRUNCATED FOR LENGTH] ...")
                break
            truncated_lines.append(line)
            current_length += len(line) + 1
            
        return '\n'.join(truncated_lines)
    
    @staticmethod
    def get_diff_summary(diff: str) -> str:
        """Get a summary of a large diff"""
        lines = diff.split('\n')
        
        files_changed = []
        added_lines = 0
        removed_lines = 0
        
        current_file = None
        for line in lines:
            if line.startswith('diff --git'):
                parts = line.split()
                if len(parts) >= 4:
                    current_file = parts[3].replace('b/', '')
                    files_changed.append(current_file)
            elif line.startswith('+') and not line.startswith('+++'):
                added_lines += 1
            elif line.startswith('-') and not line.startswith('---'):
                removed_lines += 1
                
        summary = f"Files changed: {len(files_changed)}\n"
        summary += f"Lines added: {added_lines}\n"
        summary += f"Lines removed: {removed_lines}\n"
        summary += f"Files: {', '.join(files_changed[:10])}"
        if len(files_changed) > 10:
            summary += f" ... and {len(files_changed) - 10} more"
            
        return summary

class OllamaProvider(AIProvider):
    def __init__(self):
        self.api_url = "http://localhost:11434/api/generate"

    @staticmethod
    def _get_prompt(diff: str) -> str:
        return f"""Based on the following git diff, generate a clear and concise commit message that follows conventional commits format.
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

    def generate_commit_message(self, diff: str, model: str = "llama2") -> str:
        # Ensure Ollama is ready
        ensure_ollama_ready(model)

        prompt = self._get_prompt(diff)
        
        try:
            response = requests.post(
                self.api_url,
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
            console.print("[red]Error: Failed to connect to Ollama[/red]")
            console.print(f"[red]Details: {str(e)}[/red]")
            raise

class OpenAIProvider(AIProvider):
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def _get_prompt(diff: str) -> str:
        return f"""Based on the following git diff, generate a clear and concise commit message that follows conventional commits format.
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

    def generate_commit_message(self, diff: str, model: str = "gpt-4o") -> str:
        prompt = self._get_prompt(diff)
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates clear and concise git commit messages in the conventional commits format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            console.print("[red]Error: Failed to generate message with OpenAI[/red]")
            console.print(f"[red]Details: {str(e)}[/red]")
            raise

class ClaudeProvider(AIProvider):
    def __init__(self, api_key: str):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not installed. Install with: pip install anthropic")
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate_commit_message(self, diff: str, model: str = "claude-3-5-sonnet-20241022") -> str:
        prompt = self._get_prompt(diff)
        
        # Truncate diff if too large for Claude's context window
        estimated_tokens = self.estimate_tokens(diff)
        if estimated_tokens > 180000:  # Claude's context limit with safety margin
            console.print(f"[yellow]Large diff detected ({estimated_tokens:,} tokens). Truncating...[/yellow]")
            if estimated_tokens > 300000:
                diff_summary = self.get_diff_summary(diff)
                diff = f"SUMMARY OF CHANGES:\n{diff_summary}\n\nFIRST 100 LINES OF DIFF:\n" + '\n'.join(diff.split('\n')[:100])
            else:
                diff = self.truncate_diff_intelligently(diff, max_tokens=180000)
            prompt = self._get_prompt(diff)
        
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=200,
                temperature=0.7,
                system="You are a helpful assistant that generates clear and concise git commit messages in the conventional commits format.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            console.print("[red]Error: Failed to generate message with Claude[/red]")
            console.print(f"[red]Details: {str(e)}[/red]")
            raise

    @staticmethod
    def _get_prompt(diff: str) -> str:
        return f"""Based on the following git diff, generate a clear and concise commit message that follows conventional commits format.
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