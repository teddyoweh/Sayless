#!/usr/bin/env python3

import os
import sqlite3
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading
from functools import wraps
from .config import Config

class UsageTracker:
    """Track Sayless command usage and provide analytics"""
    
    def __init__(self):
        self.config = Config()
        self.db_path = os.path.join(self.config.config_dir, 'usage.db')
        self._init_database()
        self._lock = threading.Lock()
    
    def _init_database(self):
        """Initialize the usage tracking database"""
        with sqlite3.connect(self.db_path) as conn:
            # Create the main table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS command_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command TEXT NOT NULL,
                    subcommand TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN NOT NULL,
                    execution_time REAL,
                    provider TEXT,
                    model TEXT,
                    git_repo TEXT,
                    error_message TEXT,
                    parameters TEXT,
                    user_id TEXT,
                    input_data TEXT,
                    output_data TEXT,
                    input_length INTEGER,
                    output_length INTEGER,
                    input_type TEXT
                )
            ''')
            
            # Migrate existing databases
            self._migrate_database(conn)
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_command_timestamp 
                ON command_usage(command, timestamp)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON command_usage(timestamp)
            ''')
    
    def _migrate_database(self, conn):
        """Migrate existing database to new schema"""
        cursor = conn.cursor()
        
        # Check if new columns exist
        cursor.execute("PRAGMA table_info(command_usage)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Add missing columns
        new_columns = [
            ('input_data', 'TEXT'),
            ('output_data', 'TEXT'),
            ('input_length', 'INTEGER'),
            ('output_length', 'INTEGER'),
            ('input_type', 'TEXT')
        ]
        
        for column_name, column_type in new_columns:
            if column_name not in columns:
                try:
                    cursor.execute(f'ALTER TABLE command_usage ADD COLUMN {column_name} {column_type}')
                    print(f"✅ Added column {column_name} to database")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e):
                        print(f"⚠️  Failed to add column {column_name}: {e}")
    
    def _process_data_for_storage(self, data: str, data_type: str) -> str:
        """Process input/output data for storage with privacy considerations"""
        if not data:
            return None
            
        # Truncate very long data to prevent database bloat
        MAX_LENGTH = 10000  # 10KB limit
        
        if len(data) > MAX_LENGTH:
            # For diffs, keep first and last parts
            if data_type == 'input' and '@@' in data:  # Likely a diff
                lines = data.split('\n')
                if len(lines) > 100:
                    truncated = '\n'.join(lines[:50]) + f'\n\n[... {len(lines) - 100} lines truncated ...]\n\n' + '\n'.join(lines[-50:])
                    return truncated[:MAX_LENGTH]
            
            # For other data, truncate with indicator
            return data[:MAX_LENGTH] + f"\n\n[... truncated from {len(data)} chars ...]"
        
        # Remove potential sensitive data patterns
        import re
        sensitive_patterns = [
            (r'(api[_-]?key|token|secret|password)\s*[:=]\s*["\']?([^"\'\s]+)["\']?', 
             r'\1: [REDACTED]'),
            (r'(Bearer|Authorization:\s*Bearer)\s+([A-Za-z0-9\-_]+)', 
             r'\1 [REDACTED]'),
            (r'([A-Za-z0-9+/]{20,}={0,2})', 
             lambda m: '[BASE64_REDACTED]' if len(m.group(1)) > 40 else m.group(1)),
        ]
        
        processed_data = data
        for pattern, replacement in sensitive_patterns:
            if callable(replacement):
                processed_data = re.sub(pattern, replacement, processed_data, flags=re.IGNORECASE)
            else:
                processed_data = re.sub(pattern, replacement, processed_data, flags=re.IGNORECASE)
        
        return processed_data
    
    def track_command(self, command: str, subcommand: str = None, success: bool = True, 
                     execution_time: float = None, error_message: str = None, 
                     parameters: Dict = None, input_data: str = None, output_data: str = None,
                     input_type: str = None):
        """Track a command execution with input/output data"""
        
        with self._lock:
            try:
                # Get current context
                provider = self.config.get_provider()
                model = self.config.get_model()
                
                # Get git repo context
                git_repo = None
                try:
                    from .git_ops import run_git_command
                    result = run_git_command(['rev-parse', '--show-toplevel'], check=False)
                    if result.returncode == 0:
                        git_repo = os.path.basename(result.stdout.strip())
                except:
                    pass
                
                # Get user identifier (hash of system info for privacy)
                import hashlib
                user_info = f"{os.getenv('USER', 'unknown')}-{os.uname().nodename}"
                user_id = hashlib.md5(user_info.encode()).hexdigest()[:8]
                
                # Store parameters as JSON
                params_json = json.dumps(parameters) if parameters else None
                
                # Process input/output data with privacy considerations
                processed_input = self._process_data_for_storage(input_data, 'input')
                processed_output = self._process_data_for_storage(output_data, 'output')
                
                # Calculate lengths (original data length for stats)
                input_length = len(input_data) if input_data else 0
                output_length = len(output_data) if output_data else 0
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT INTO command_usage 
                        (command, subcommand, success, execution_time, provider, model, 
                         git_repo, error_message, parameters, user_id, input_data, output_data,
                         input_length, output_length, input_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (command, subcommand, success, execution_time, provider, model,
                          git_repo, error_message, params_json, user_id, processed_input,
                          processed_output, input_length, output_length, input_type))
            except Exception as e:
                # Don't let tracking failures break the main functionality
                pass
    
    def get_usage_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        
        since_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            stats = {
                'total_commands': 0,
                'success_rate': 0.0,
                'avg_execution_time': 0.0,
                'most_used_commands': [],
                'provider_distribution': {},
                'daily_usage': [],
                'recent_activity': [],
                'error_summary': [],
                'command_success_rates': {},
                'performance_metrics': {},
                'input_output_analytics': {}
            }
            
            # Total commands and success rate
            result = conn.execute('''
                SELECT COUNT(*) as total, 
                       AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                       AVG(execution_time) as avg_time
                FROM command_usage 
                WHERE timestamp >= ?
            ''', (since_date,)).fetchone()
            
            if result:
                stats['total_commands'] = result['total']
                stats['success_rate'] = result['success_rate'] or 0.0
                stats['avg_execution_time'] = result['avg_time'] or 0.0
            
            # Most used commands
            commands = conn.execute('''
                SELECT command, subcommand, COUNT(*) as count,
                       AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate
                FROM command_usage 
                WHERE timestamp >= ?
                GROUP BY command, subcommand
                ORDER BY count DESC
                LIMIT 10
            ''', (since_date,)).fetchall()
            
            stats['most_used_commands'] = [
                {
                    'command': f"{row['command']}{(' ' + row['subcommand']) if row['subcommand'] else ''}",
                    'count': row['count'],
                    'success_rate': row['success_rate']
                }
                for row in commands
            ]
            
            # Provider distribution
            providers = conn.execute('''
                SELECT provider, COUNT(*) as count
                FROM command_usage 
                WHERE timestamp >= ? AND provider IS NOT NULL
                GROUP BY provider
                ORDER BY count DESC
            ''', (since_date,)).fetchall()
            
            stats['provider_distribution'] = {
                row['provider']: row['count'] for row in providers
            }
            
            # Daily usage for the last 14 days
            daily_usage = conn.execute('''
                SELECT DATE(timestamp) as date, COUNT(*) as count,
                       AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate
                FROM command_usage 
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
                LIMIT 14
            ''', (datetime.now() - timedelta(days=14),)).fetchall()
            
            stats['daily_usage'] = [
                {
                    'date': row['date'],
                    'count': row['count'],
                    'success_rate': row['success_rate']
                }
                for row in reversed(daily_usage)  # Reverse to get chronological order
            ]
            
            # Recent activity
            recent = conn.execute('''
                SELECT command, subcommand, timestamp, success, execution_time, provider
                FROM command_usage 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 20
            ''', (since_date,)).fetchall()
            
            stats['recent_activity'] = [
                {
                    'command': f"{row['command']}{(' ' + row['subcommand']) if row['subcommand'] else ''}",
                    'timestamp': row['timestamp'],
                    'success': row['success'],
                    'execution_time': row['execution_time'],
                    'provider': row['provider']
                }
                for row in recent
            ]
            
            # Error summary
            errors = conn.execute('''
                SELECT command, COUNT(*) as count, error_message
                FROM command_usage 
                WHERE timestamp >= ? AND success = 0 AND error_message IS NOT NULL
                GROUP BY command, error_message
                ORDER BY count DESC
                LIMIT 5
            ''', (since_date,)).fetchall()
            
            stats['error_summary'] = [
                {
                    'command': row['command'],
                    'count': row['count'],
                    'error': row['error_message'][:100] + '...' if len(row['error_message']) > 100 else row['error_message']
                }
                for row in errors
            ]
            
            # Command success rates
            cmd_success = conn.execute('''
                SELECT command, 
                       COUNT(*) as total,
                       AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                       AVG(execution_time) as avg_time
                FROM command_usage 
                WHERE timestamp >= ?
                GROUP BY command
                HAVING total >= 3
                ORDER BY success_rate ASC
            ''', (since_date,)).fetchall()
            
            stats['command_success_rates'] = {
                row['command']: {
                    'success_rate': row['success_rate'],
                    'total': row['total'],
                    'avg_time': row['avg_time']
                }
                for row in cmd_success
            }
            
            # Performance metrics
            perf = conn.execute('''
                SELECT command,
                       AVG(execution_time) as avg_time,
                       MIN(execution_time) as min_time,
                       MAX(execution_time) as max_time
                FROM command_usage 
                WHERE timestamp >= ? AND execution_time IS NOT NULL
                GROUP BY command
                ORDER BY avg_time DESC
            ''', (since_date,)).fetchall()
            
            stats['performance_metrics'] = {
                row['command']: {
                    'avg_time': row['avg_time'],
                    'min_time': row['min_time'],
                    'max_time': row['max_time']
                }
                for row in perf
            }
            
            # Input/Output Analytics
            io_analytics = conn.execute('''
                SELECT 
                    COUNT(*) as total_with_io,
                    AVG(input_length) as avg_input_length,
                    AVG(output_length) as avg_output_length,
                    MAX(input_length) as max_input_length,
                    MAX(output_length) as max_output_length,
                    input_type,
                    COUNT(DISTINCT command) as commands_with_io
                FROM command_usage 
                WHERE timestamp >= ? AND (input_data IS NOT NULL OR output_data IS NOT NULL)
                GROUP BY input_type
            ''', (since_date,)).fetchall()
            
            # Input/Output by command
            io_by_command = conn.execute('''
                SELECT 
                    command,
                    COUNT(*) as total_io,
                    AVG(input_length) as avg_input_length,
                    AVG(output_length) as avg_output_length,
                    input_type
                FROM command_usage 
                WHERE timestamp >= ? AND (input_data IS NOT NULL OR output_data IS NOT NULL)
                GROUP BY command, input_type
                ORDER BY total_io DESC
            ''', (since_date,)).fetchall()
            
            # Recent input/output samples (preview)
            recent_io = conn.execute('''
                SELECT 
                    command,
                    input_type,
                    input_length,
                    output_length,
                    SUBSTR(input_data, 1, 200) as input_preview,
                    SUBSTR(output_data, 1, 200) as output_preview,
                    timestamp
                FROM command_usage 
                WHERE timestamp >= ? AND (input_data IS NOT NULL OR output_data IS NOT NULL)
                ORDER BY timestamp DESC
                LIMIT 10
            ''', (since_date,)).fetchall()
            
            stats['input_output_analytics'] = {
                'summary': {
                    'total_commands_with_io': sum(row['total_with_io'] for row in io_analytics),
                    'avg_input_length': sum(row['avg_input_length'] or 0 for row in io_analytics) / len(io_analytics) if io_analytics else 0,
                    'avg_output_length': sum(row['avg_output_length'] or 0 for row in io_analytics) / len(io_analytics) if io_analytics else 0,
                    'max_input_length': max((row['max_input_length'] or 0 for row in io_analytics), default=0),
                    'max_output_length': max((row['max_output_length'] or 0 for row in io_analytics), default=0),
                    'input_types': [row['input_type'] for row in io_analytics if row['input_type']]
                },
                'by_command': [
                    {
                        'command': row['command'],
                        'total_io': row['total_io'],
                        'avg_input_length': row['avg_input_length'],
                        'avg_output_length': row['avg_output_length'],
                        'input_type': row['input_type']
                    }
                    for row in io_by_command
                ],
                'recent_samples': [
                    {
                        'command': row['command'],
                        'input_type': row['input_type'],
                        'input_length': row['input_length'],
                        'output_length': row['output_length'],
                        'input_preview': row['input_preview'],
                        'output_preview': row['output_preview'],
                        'timestamp': row['timestamp']
                    }
                    for row in recent_io
                ]
            }
            
            return stats
    
    def get_command_trends(self, command: str, days: int = 30) -> Dict[str, Any]:
        """Get trends for a specific command"""
        
        since_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Daily usage trend
            daily_trend = conn.execute('''
                SELECT DATE(timestamp) as date, 
                       COUNT(*) as count,
                       AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                       AVG(execution_time) as avg_time
                FROM command_usage 
                WHERE command = ? AND timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            ''', (command, since_date)).fetchall()
            
            return {
                'command': command,
                'daily_trend': [
                    {
                        'date': row['date'],
                        'count': row['count'],
                        'success_rate': row['success_rate'],
                        'avg_time': row['avg_time']
                    }
                    for row in daily_trend
                ]
            }
    
    def export_usage_data(self, format: str = 'json', days: int = 30) -> str:
        """Export usage data in various formats"""
        
        stats = self.get_usage_stats(days)
        
        if format == 'json':
            return json.dumps(stats, indent=2, default=str)
        elif format == 'csv':
            # Simple CSV export of raw data
            since_date = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                results = conn.execute('''
                    SELECT * FROM command_usage 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                ''', (since_date,)).fetchall()
                
                import csv
                import io
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                if results:
                    writer.writerow(results[0].keys())
                    
                    # Write data
                    for row in results:
                        writer.writerow(row)
                
                return output.getvalue()
        
        return json.dumps(stats, indent=2, default=str)
    
    def get_detailed_io_samples(self, days: int = 30, limit: int = 100, command: str = None, input_type: str = None) -> List[Dict[str, Any]]:
        """Get detailed input/output samples with filtering"""
        
        since_date = datetime.now() - timedelta(days=days)
        
        # Build query with filters
        query = '''
            SELECT 
                command,
                subcommand,
                input_type,
                input_length,
                output_length,
                input_data,
                output_data,
                SUBSTR(input_data, 1, 500) as input_preview,
                SUBSTR(output_data, 1, 500) as output_preview,
                timestamp,
                execution_time,
                success,
                provider,
                model
            FROM command_usage 
            WHERE timestamp >= ? AND (input_data IS NOT NULL OR output_data IS NOT NULL)
        '''
        
        params = [since_date]
        
        # Add command filter
        if command:
            query += ' AND command = ?'
            params.append(command)
        
        # Add input type filter
        if input_type:
            query += ' AND input_type = ?'
            params.append(input_type)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            results = conn.execute(query, params).fetchall()
            
            samples = []
            for row in results:
                # Create full command name
                full_command = row['command']
                if row['subcommand']:
                    full_command += f" {row['subcommand']}"
                
                sample = {
                    'command': full_command,
                    'input_type': row['input_type'],
                    'input_length': row['input_length'] or 0,
                    'output_length': row['output_length'] or 0,
                    'input_preview': row['input_preview'],
                    'output_preview': row['output_preview'],
                    'input_data': row['input_data'],  # Full data for modal view
                    'output_data': row['output_data'], # Full data for modal view
                    'timestamp': row['timestamp'],
                    'execution_time': row['execution_time'],
                    'success': row['success'],
                    'provider': row['provider'],
                    'model': row['model']
                }
                samples.append(sample)
            
            return samples


# Global tracker instance
_tracker = None

def get_tracker() -> UsageTracker:
    """Get the global usage tracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = UsageTracker()
    return _tracker


def track_command_usage(command_name: str, subcommand: str = None):
    """Decorator to track command usage"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error_message = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                execution_time = time.time() - start_time
                
                # Extract parameters from kwargs for tracking
                params = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, bool, float))}
                
                tracker = get_tracker()
                tracker.track_command(
                    command=command_name,
                    subcommand=subcommand,
                    success=success,
                    execution_time=execution_time,
                    error_message=error_message,
                    parameters=params
                )
        
        return wrapper
    return decorator


def track_command_manual(command: str, subcommand: str = None, success: bool = True, 
                        execution_time: float = None, error_message: str = None, 
                        parameters: Dict = None, input_data: str = None, output_data: str = None,
                        input_type: str = None):
    """Manually track a command execution with input/output data"""
    tracker = get_tracker()
    tracker.track_command(
        command=command,
        subcommand=subcommand,
        success=success,
        execution_time=execution_time,
        error_message=error_message,
        parameters=parameters,
        input_data=input_data,
        output_data=output_data,
        input_type=input_type
    ) 