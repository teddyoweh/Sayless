import os
import json
from pathlib import Path
from rich.console import Console

console = Console()

class Config:
    def __init__(self):
        self.config_dir = Path.home() / '.sayless'
        self.config_file = self.config_dir / 'config.json'
        self.config = self.load_config()

    def load_config(self):
        """Load configuration from file or create default"""
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True)
        
        if not self.config_file.exists():
            default_config = {
                'provider': 'openai',  # 'openai' or 'ollama'
                'openai_api_key': None,
                'model': 'gpt-4o'  # default model for OpenAI
            }
            self.save_config(default_config)
            return default_config
        
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            console.print("[red]Error: Invalid config file. Resetting to defaults.[/red]")
            return self.reset_config()

    def save_config(self, config):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        self.config = config

    def reset_config(self):
        """Reset configuration to defaults"""
        default_config = {
            'provider': 'openai',
            'openai_api_key': None,
            'model': 'gpt-4o'
        }
        self.save_config(default_config)
        return default_config

    def get_openai_api_key(self):
        """Get OpenAI API key from config or environment"""
        # First check environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            return api_key
        
        # Then check config file
        return self.config.get('openai_api_key')

    def set_openai_api_key(self, api_key):
        """Set OpenAI API key in config and environment"""
        self.config['openai_api_key'] = api_key
        self.save_config(self.config)
        # Also set it for the current session
        os.environ['OPENAI_API_KEY'] = api_key

    def set_provider(self, provider):
        """Set the AI provider (openai or ollama)"""
        if provider not in ['ollama', 'openai']:
            raise ValueError("Provider must be 'openai' or 'ollama'")
        self.config['provider'] = provider
        # Set appropriate default model when switching providers
        if provider == 'openai' and self.config.get('model') == 'llama2':
            self.config['model'] = 'gpt-4o'
        elif provider == 'ollama' and 'gpt' in self.config.get('model', ''):
            self.config['model'] = 'llama2'
        self.save_config(self.config)

    def get_provider(self):
        """Get current AI provider"""
        return self.config.get('provider', 'openai')

    def set_model(self, model):
        """Set the model name"""
        self.config['model'] = model
        self.save_config(self.config)

    def get_model(self):
        """Get current model name"""
        return self.config.get('model', 'gpt-4o') 