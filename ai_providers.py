from abc import ABC, abstractmethod
import requests
from openai import OpenAI
from rich.console import Console
from setup import ensure_ollama_ready

console = Console()

class AIProvider(ABC):
    @abstractmethod
    def generate_commit_message(self, diff: str, model: str) -> str:
        pass

class OllamaProvider(AIProvider):
    def __init__(self):
        self.api_url = "http://localhost:11434/api/generate"

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