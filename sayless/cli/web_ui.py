#!/usr/bin/env python3

import os
import json
import subprocess
from flask import Flask, render_template, request, jsonify, redirect, url_for, make_response
from werkzeug.serving import make_server
import threading
import webbrowser
from typing import Dict, List, Any
import time
from pathlib import Path

from .config import Config
from .ai_providers import OpenAIProvider, OllamaProvider, ClaudeProvider
from .git_ops import run_git_command
from .github_ops import GitHubAPI
from .usage_tracker import get_tracker

class SaylessWebUI:
    """Modern web UI for Sayless configuration and management"""
    
    def __init__(self):
        self.app = Flask(__name__, 
                        template_folder=str(Path(__file__).parent / 'templates'),
                        static_folder=str(Path(__file__).parent / 'static'))
        self.config = Config()
        self.server = None
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard"""
            return render_template('index.html', 
                                 status=self.get_system_status(),
                                 config=self.get_config_data())
        
        @self.app.route('/api/status')
        def api_status():
            """API endpoint for system status"""
            return jsonify(self.get_system_status())
        
        @self.app.route('/api/config', methods=['GET', 'POST'])
        def api_config():
            """API endpoint for configuration"""
            if request.method == 'GET':
                return jsonify(self.get_config_data())
            
            elif request.method == 'POST':
                data = request.get_json()
                success = self.update_config(data)
                return jsonify({'success': success, 'config': self.get_config_data()})
        
        @self.app.route('/api/test-connection')
        def api_test_connection():
            """Test AI provider connections"""
            provider = request.args.get('provider', 'openai')
            result = self.test_ai_connection(provider)
            return jsonify(result)
        
        @self.app.route('/api/commands')
        def api_commands():
            """Get command documentation"""
            return jsonify(self.get_command_docs())
        
        @self.app.route('/help')
        def help_page():
            """Help and documentation page"""
            return render_template('help.html', commands=self.get_command_docs())
        
        @self.app.route('/settings')
        def settings_page():
            """Settings configuration page"""
            return render_template('settings.html', 
                                 config=self.get_config_data(),
                                 status=self.get_system_status())
        
        @self.app.route('/favicon.ico')
        def favicon():
            """Serve the logo as favicon"""
            return self.app.send_static_file('logo.png')
        
        @self.app.route('/analytics')
        def analytics_page():
            """Analytics and usage tracking page"""
            return render_template('analytics.html')
        
        @self.app.route('/io-samples')
        def io_samples_page():
            """Input/Output samples page"""
            return render_template('io-samples.html')
        
        @self.app.route('/api/analytics/overview')
        def api_analytics_overview():
            """Get usage analytics overview"""
            days = request.args.get('days', 30, type=int)
            tracker = get_tracker()
            stats = tracker.get_usage_stats(days)
            return jsonify(stats)
        
        @self.app.route('/api/analytics/io-samples')
        def api_analytics_io_samples():
            """Get input/output samples for detailed analysis"""
            days = request.args.get('days', 30, type=int)
            limit = request.args.get('limit', 100, type=int)
            command = request.args.get('command', None)
            input_type = request.args.get('input_type', None)
            
            tracker = get_tracker()
            samples = tracker.get_detailed_io_samples(days, limit, command, input_type)
            return jsonify({'samples': samples})
        
        @self.app.route('/api/analytics/command/<command>')
        def api_analytics_command(command):
            """Get analytics for a specific command"""
            days = request.args.get('days', 30, type=int)
            tracker = get_tracker()
            trends = tracker.get_command_trends(command, days)
            return jsonify(trends)
        
        @self.app.route('/api/analytics/export')
        def api_analytics_export():
            """Export usage data"""
            format_type = request.args.get('format', 'json')
            days = request.args.get('days', 30, type=int)
            tracker = get_tracker()
            
            data = tracker.export_usage_data(format_type, days)
            
            if format_type == 'csv':
                response = make_response(data)
                response.headers['Content-Type'] = 'text/csv'
                response.headers['Content-Disposition'] = f'attachment; filename=sayless_usage_{days}days.csv'
                return response
            else:
                return jsonify({'data': data})
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'provider': self.config.get_provider(),
            'model': self.config.get_model(),
            'openai_configured': bool(self.config.get_openai_api_key()),
            'claude_configured': bool(self.config.get_claude_api_key()),
            'github_configured': bool(self.config.get_github_token()),
            'git_repo': False,
            'git_status': None,
            'ollama_available': False,
            'version': '0.2.0'
        }
        
        # Check git repository
        try:
            run_git_command(['rev-parse', '--git-dir'], check=True)
            status['git_repo'] = True
            
            # Get git status
            result = run_git_command(['status', '--porcelain'], check=False)
            if result.returncode == 0:
                changes = result.stdout.strip()
                status['git_status'] = {
                    'clean': not bool(changes),
                    'changes': len(changes.split('\n')) if changes else 0
                }
        except:
            pass
        
        # Check Ollama availability
        try:
            import requests
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.status_code == 200:
                status['ollama_available'] = True
                status['ollama_models'] = [model['name'] for model in response.json().get('models', [])]
        except:
            status['ollama_models'] = []
        
        # Test connections
        if status['openai_configured']:
            status['openai_working'] = self.test_openai_connection()
        
        if status['claude_configured']:
            status['claude_working'] = self.test_claude_connection()
        
        if status['github_configured']:
            status['github_working'] = self.test_github_connection()
        
        return status
    
    def get_config_data(self) -> Dict[str, Any]:
        """Get current configuration data"""
        return {
            'provider': self.config.get_provider(),
            'model': self.config.get_model(),
            'openai_key_set': bool(self.config.get_openai_api_key()),
            'claude_key_set': bool(self.config.get_claude_api_key()),
            'github_token_set': bool(self.config.get_github_token()),
            'config_path': str(self.config.config_file)
        }
    
    def update_config(self, data: Dict[str, Any]) -> bool:
        """Update configuration with new data"""
        try:
            if 'provider' in data:
                self.config.set_provider(data['provider'])
            
            if 'model' in data:
                self.config.set_model(data['model'])
            
            if 'openai_key' in data and data['openai_key']:
                self.config.set_openai_api_key(data['openai_key'])
            
            if 'claude_key' in data and data['claude_key']:
                self.config.set_claude_api_key(data['claude_key'])
            
            if 'github_token' in data and data['github_token']:
                self.config.set_github_token(data['github_token'])
            
            return True
        except Exception as e:
            print(f"Error updating config: {e}")
            return False
    
    def test_openai_connection(self) -> bool:
        """Test OpenAI connection"""
        try:
            api_key = self.config.get_openai_api_key()
            if not api_key:
                return False
            
            provider = OpenAIProvider(api_key)
            # Test with a simple prompt
            provider.generate_commit_message("test", "gpt-4o-mini")
            return True
        except:
            return False
    
    def test_claude_connection(self) -> bool:
        """Test Claude connection"""
        try:
            api_key = self.config.get_claude_api_key()
            if not api_key:
                return False
            
            provider = ClaudeProvider(api_key)
            # Test with a simple prompt
            provider.generate_commit_message("test", "claude-3-5-sonnet-20241022")
            return True
        except:
            return False
    
    def test_github_connection(self) -> bool:
        """Test GitHub connection"""
        try:
            github = GitHubAPI()
            # Test by getting user info
            import requests
            response = requests.get(f"{github.api_url}/user", headers=github.headers)
            return response.status_code == 200
        except:
            return False
    
    def test_ai_connection(self, provider: str) -> Dict[str, Any]:
        """Test AI provider connection"""
        if provider == 'openai':
            success = self.test_openai_connection()
            return {
                'success': success,
                'message': 'OpenAI connection successful' if success else 'OpenAI connection failed - check API key'
            }
        elif provider == 'claude':
            success = self.test_claude_connection()
            return {
                'success': success,
                'message': 'Claude connection successful' if success else 'Claude connection failed - check API key'
            }
        elif provider == 'ollama':
            try:
                import requests
                response = requests.get('http://localhost:11434/api/tags', timeout=5)
                success = response.status_code == 200
                return {
                    'success': success,
                    'message': 'Ollama connection successful' if success else 'Ollama not running - start Ollama first'
                }
            except:
                return {
                    'success': False,
                    'message': 'Ollama not available - install and start Ollama'
                }
        
        return {'success': False, 'message': 'Unknown provider'}
    
    def get_command_docs(self) -> List[Dict[str, Any]]:
        """Get comprehensive command documentation"""
        commands = [
            {
                'name': 'generate',
                'alias': 'g',
                'description': 'Generate AI-powered commit messages',
                'usage': 'sayless generate [--preview] [-a/--auto-add]',
                'examples': [
                    'sayless g',
                    'sayless g -a --preview',
                    'sayless generate --auto-add'
                ],
                'options': [
                    {'flag': '--preview', 'description': 'Preview commit message without creating commit'},
                    {'flag': '-a, --auto-add', 'description': 'Automatically stage all changes before generating commit'}
                ],
                'category': 'Core'
            },
            {
                'name': 'branch',
                'alias': None,
                'description': 'Create AI-generated branch names',
                'usage': 'sayless branch [description] [--generate/-g] [--no-checkout] [-a]',
                'examples': [
                    'sayless branch "add user authentication"',
                    'sayless branch -g',
                    'sayless branch -g -a'
                ],
                'options': [
                    {'flag': '-g, --generate', 'description': 'Generate branch name from staged changes'},
                    {'flag': '--no-checkout', 'description': 'Create branch without switching to it'},
                    {'flag': '-a', 'description': 'Auto-stage changes before operation'}
                ],
                'category': 'Git'
            },
            {
                'name': 'pr create',
                'alias': None,
                'description': 'Create pull requests with AI-generated content',
                'usage': 'sayless pr create [--base branch] [--no-push]',
                'examples': [
                    'sayless pr create',
                    'sayless pr create --base develop',
                    'sayless pr create --no-push'
                ],
                'options': [
                    {'flag': '--base', 'description': 'Specify base branch (default: main)'},
                    {'flag': '--no-push', 'description': 'Don\'t automatically push branch'}
                ],
                'category': 'GitHub'
            },
            {
                'name': 'review-enhanced',
                'alias': None,
                'description': 'Structured AI-powered PR reviews',
                'usage': 'sayless review-enhanced [--pr number] [--type type] [--auto-post]',
                'examples': [
                    'sayless review-enhanced --pr 123',
                    'sayless review-enhanced --pr 123 --type security',
                    'sayless review-enhanced --current --type detailed'
                ],
                'options': [
                    {'flag': '--pr', 'description': 'PR number to review'},
                    {'flag': '--type', 'description': 'Review type: quick, detailed, security, performance, dependencies'},
                    {'flag': '--auto-post', 'description': 'Automatically post review to GitHub'},
                    {'flag': '--current', 'description': 'Review current branch'}
                ],
                'category': 'Review'
            },
            {
                'name': 'bulk-review',
                'alias': None,
                'description': 'Review multiple PRs at once',
                'usage': 'sayless bulk-review [--type type] [--max number] [--auto-post]',
                'examples': [
                    'sayless bulk-review',
                    'sayless bulk-review --type security --max 10',
                    'sayless bulk-review --auto-post'
                ],
                'options': [
                    {'flag': '--type', 'description': 'Review type for all PRs'},
                    {'flag': '--max', 'description': 'Maximum number of PRs to review'},
                    {'flag': '--auto-post', 'description': 'Auto-post all reviews'}
                ],
                'category': 'Review'
            },
            {
                'name': 'deps analyze',
                'alias': 'analyze-deps',
                'description': 'Analyze and manage dependencies',
                'usage': 'sayless deps analyze [--auto-fix] [--ecosystem type]',
                'examples': [
                    'sayless deps analyze',
                    'sayless deps analyze --auto-fix',
                    'sayless analyze-deps --auto-fix'
                ],
                'options': [
                    {'flag': '--auto-fix', 'description': 'Automatically add missing dependencies'},
                    {'flag': '--ecosystem', 'description': 'Focus on specific ecosystem (npm, pip, poetry)'}
                ],
                'category': 'Dependencies'
            },
            {
                'name': 'search',
                'alias': None,
                'description': 'Search commits using AI-powered semantic search',
                'usage': 'sayless search "query" [--limit number] [--index-all]',
                'examples': [
                    'sayless search "authentication bug"',
                    'sayless search "database optimization" --limit 10',
                    'sayless search "api changes" --index-all'
                ],
                'options': [
                    {'flag': '--limit', 'description': 'Maximum number of results'},
                    {'flag': '--index-all', 'description': 'Re-index all commits before searching'}
                ],
                'category': 'Search'
            },
            {
                'name': 'config',
                'alias': None,
                'description': 'Configure AI providers and settings',
                'usage': 'sayless config [--show] [--openai-key key] [--claude-key key] [--github-token token]',
                'examples': [
                    'sayless config --show',
                    'sayless config --openai-key YOUR_KEY',
                    'sayless config --claude-key YOUR_KEY',
                    'sayless config --github-token YOUR_TOKEN'
                ],
                'options': [
                    {'flag': '--show', 'description': 'Show current configuration'},
                    {'flag': '--openai-key', 'description': 'Set OpenAI API key'},
                    {'flag': '--claude-key', 'description': 'Set Claude API key'},
                    {'flag': '--github-token', 'description': 'Set GitHub token'}
                ],
                'category': 'Configuration'
            },
            {
                'name': 'switch',
                'alias': None,
                'description': 'Switch between AI providers',
                'usage': 'sayless switch provider [--key key] [--model model]',
                'examples': [
                    'sayless switch openai --key YOUR_KEY',
                    'sayless switch claude --key YOUR_KEY',
                    'sayless switch ollama',
                    'sayless switch openai --model gpt-4o'
                ],
                'options': [
                    {'flag': '--key', 'description': 'API key for OpenAI/Claude'},
                    {'flag': '--model', 'description': 'Model to use'}
                ],
                'category': 'Configuration'
            },
            {
                'name': 'setup',
                'alias': None,
                'description': 'Launch beautiful web-based setup and configuration interface',
                'usage': 'sayless setup [--port number] [--host address] [--no-browser]',
                'examples': [
                    'sayless setup',
                    'sayless setup --port 3000',
                    'sayless setup --host 0.0.0.0 --port 8080',
                    'sayless setup --no-browser'
                ],
                'options': [
                    {'flag': '--port, -p', 'description': 'Port to run the web UI on (default: 8888)'},
                    {'flag': '--host', 'description': 'Host to bind the web UI to (default: 127.0.0.1)'},
                    {'flag': '--no-browser', 'description': 'Don\'t automatically open browser'}
                ],
                'category': 'Configuration'
            }
        ]
        
        return commands
    
    def start_server(self, host='127.0.0.1', port=8888, open_browser=True):
        """Start the Flask development server"""
        def run_server():
            self.server = make_server(host, port, self.app, threaded=True)
            self.server.serve_forever()
        
        # Start server in background thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait a moment for server to start
        time.sleep(1)
        
        url = f"http://{host}:{port}"
        print(f"🚀 Sayless Setup UI running at: {url}")
        
        if open_browser:
            try:
                webbrowser.open(url)
                print("📱 Opening in your default browser...")
            except:
                print("💡 Open the URL above in your browser")
        
        return url
    
    def stop_server(self):
        """Stop the Flask server"""
        if self.server:
            self.server.shutdown()
            print("🛑 Setup UI stopped")

def create_ui_directories():
    """Create necessary directories for templates and static files"""
    base_dir = Path(__file__).parent
    templates_dir = base_dir / 'templates'
    static_dir = base_dir / 'static'
    
    templates_dir.mkdir(exist_ok=True)
    static_dir.mkdir(exist_ok=True)
    
    return templates_dir, static_dir

def launch_setup_ui(host='127.0.0.1', port=8888, open_browser=True):
    """Launch the Sayless setup UI"""
    # Create directories
    create_ui_directories()
    
    # Create and start UI
    ui = SaylessWebUI()
    url = ui.start_server(host, port, open_browser)
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ui.stop_server()
        print("\n👋 Setup UI closed")
    
    return ui 