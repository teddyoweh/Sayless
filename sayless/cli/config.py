import os
import json
from pathlib import Path
from rich.console import Console

console = Console()

class Config:
    def __init__(self):
        """Initialize configuration"""
        self.config_dir = os.path.expanduser("~/.sayless")
        self.config_file = os.path.join(self.config_dir, "config.json")
        
        # Create config directory if it doesn't exist
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
        
        # Load or create config file
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'provider': 'openai',
                'model': 'gpt-4o',
                'openai_api_key': None,
                'github_token': None
            }
            self.save_config(self.config)

    def save_config(self, config):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)

    def get_openai_api_key(self):
        """Get OpenAI API key from config or environment"""
        return os.getenv('OPENAI_API_KEY') or self.config.get('openai_api_key')

    def set_openai_api_key(self, api_key):
        """Set OpenAI API key in config and environment"""
        self.config['openai_api_key'] = api_key
        self.save_config(self.config)
        # Also set it for the current session
        os.environ['OPENAI_API_KEY'] = api_key

    def get_github_token(self):
        """Get GitHub token from config or environment"""
        return os.getenv('GITHUB_TOKEN') or self.config.get('github_token')

    def set_github_token(self, token):
        """Set GitHub token in config and environment"""
        self.config['github_token'] = token
        self.save_config(self.config)
        # Also set it for the current session
        os.environ['GITHUB_TOKEN'] = token

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