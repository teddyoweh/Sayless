import subprocess
import json
import re
import os
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import requests
from dataclasses import dataclass
from .ai_providers import OpenAIProvider, OllamaProvider
from .config import Config
from .git_ops import run_git_command

@dataclass
class DetectedDependency:
    """Represents a dependency detected from code analysis"""
    package_name: str
    import_statement: str
    file_path: str
    line_number: int
    ecosystem: str  # 'npm', 'pip', 'poetry'
    confidence: float  # 0.0 to 1.0
    suggested_version: Optional[str] = None

console = Console()
settings = Config()

@dataclass
class DependencyChange:
    """Represents a change in a dependency"""
    name: str
    old_version: Optional[str]
    new_version: Optional[str]
    change_type: str  # 'added', 'removed', 'updated', 'downgraded'
    ecosystem: str  # 'npm', 'pip', 'poetry'
    breaking_changes: List[str]
    security_impact: Optional[str]
    
@dataclass
class DependencyFile:
    """Represents a dependency file in the repository"""
    path: str
    ecosystem: str
    dependencies: Dict[str, str]
    dev_dependencies: Dict[str, str] = None

class DependencyManager:
    """Manages intelligent dependency evolution and analysis"""
    
    def __init__(self):
        self.repo_root = self._find_repo_root()
        self.dependency_files = self._detect_dependency_files()
        
    def _find_repo_root(self) -> str:
        """Find the root of the git repository"""
        try:
            result = run_git_command(['rev-parse', '--show-toplevel'])
            return result.stdout.strip()
        except:
            return os.getcwd()
    
    def _detect_dependency_files(self) -> List[DependencyFile]:
        """Detect all dependency files in the repository"""
        files = []
        repo_path = Path(self.repo_root)
        
        # Node.js files
        for package_json in repo_path.rglob('package.json'):
            if 'node_modules' not in str(package_json):
                try:
                    with open(package_json, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        files.append(DependencyFile(
                            path=str(package_json.relative_to(repo_path)),
                            ecosystem='npm',
                            dependencies=data.get('dependencies', {}),
                            dev_dependencies=data.get('devDependencies', {})
                        ))
                except Exception:
                    continue
        
        # Python files
        for req_file in repo_path.rglob('requirements*.txt'):
            try:
                dependencies = self._parse_requirements_txt(req_file)
                files.append(DependencyFile(
                    path=str(req_file.relative_to(repo_path)),
                    ecosystem='pip',
                    dependencies=dependencies
                ))
            except Exception:
                continue
        
        # Python Poetry files
        for pyproject in repo_path.rglob('pyproject.toml'):
            try:
                dependencies = self._parse_pyproject_toml(pyproject)
                if dependencies:  # Only add if it has dependencies
                    files.append(DependencyFile(
                        path=str(pyproject.relative_to(repo_path)),
                        ecosystem='poetry',
                        dependencies=dependencies.get('dependencies', {}),
                        dev_dependencies=dependencies.get('dev_dependencies', {})
                    ))
            except Exception:
                continue
                
        return files
    
    def _parse_requirements_txt(self, file_path: Path) -> Dict[str, str]:
        """Parse requirements.txt file"""
        dependencies = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('-'):
                        # Handle various formats: package==1.0.0, package>=1.0.0, package
                        match = re.match(r'^([a-zA-Z0-9\-_\.]+)([><=!]+)?(.+)?', line)
                        if match:
                            name = match.group(1)
                            operator = match.group(2) or '=='
                            version = match.group(3) or 'latest'
                            dependencies[name] = f"{operator}{version}" if operator != '==' else version
        except Exception:
            pass
        return dependencies
    
    def _parse_pyproject_toml(self, file_path: Path) -> Optional[Dict]:
        """Parse pyproject.toml file (basic parsing without toml library)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Simple regex-based parsing for dependencies
            dependencies = {}
            dev_dependencies = {}
            
            # Look for [tool.poetry.dependencies] section
            deps_match = re.search(r'\[tool\.poetry\.dependencies\](.*?)(?=\[|$)', content, re.DOTALL)
            if deps_match:
                deps_section = deps_match.group(1)
                for line in deps_section.split('\n'):
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        parts = line.split('=', 1)
                        if len(parts) == 2:
                            name = parts[0].strip()
                            version = parts[1].strip().strip('"\'')
                            if name != 'python':  # Skip python version requirement
                                dependencies[name] = version
            
            # Look for [tool.poetry.group.dev.dependencies] section
            dev_deps_match = re.search(r'\[tool\.poetry\.group\.dev\.dependencies\](.*?)(?=\[|$)', content, re.DOTALL)
            if dev_deps_match:
                dev_deps_section = dev_deps_match.group(1)
                for line in dev_deps_section.split('\n'):
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        parts = line.split('=', 1)
                        if len(parts) == 2:
                            name = parts[0].strip()
                            version = parts[1].strip().strip('"\'')
                            dev_dependencies[name] = version
            
            if dependencies or dev_dependencies:
                return {
                    'dependencies': dependencies,
                    'dev_dependencies': dev_dependencies
                }
        except Exception:
            pass
        return None
    
    def analyze_commit_dependencies(self, commit_hash: str = None) -> List[DependencyChange]:
        """Analyze dependency changes in a specific commit or current changes"""
        changes = []
        
        try:
            if commit_hash:
                # Get diff for specific commit
                diff_cmd = ['show', '--name-only', commit_hash]
            else:
                # Get current staged/unstaged changes
                diff_cmd = ['diff', '--name-only', 'HEAD']
            
            changed_files = run_git_command(diff_cmd).stdout.strip().split('\n')
            changed_files = [f for f in changed_files if f.strip()]
            
            # Filter for dependency files
            dep_file_patterns = [
                'package.json', 'requirements.txt', 'requirements-dev.txt', 
                'pyproject.toml', 'poetry.lock', 'package-lock.json', 'yarn.lock'
            ]
            
            changed_dep_files = []
            for file in changed_files:
                for pattern in dep_file_patterns:
                    if pattern in file:
                        changed_dep_files.append(file)
                        break
            
            # Analyze each changed dependency file
            for file in changed_dep_files:
                file_changes = self._analyze_file_changes(file, commit_hash)
                changes.extend(file_changes)
                
        except Exception as e:
            console.print(f"[yellow]Warning: Could not analyze dependencies: {str(e)}[/yellow]")
        
        return changes
    
    def _analyze_file_changes(self, file_path: str, commit_hash: str = None) -> List[DependencyChange]:
        """Analyze changes in a specific dependency file"""
        changes = []
        
        try:
            if commit_hash:
                # Get before and after content for specific commit
                old_content = self._get_file_content_at_commit(file_path, f"{commit_hash}^")
                new_content = self._get_file_content_at_commit(file_path, commit_hash)
            else:
                # Get current vs HEAD content
                old_content = self._get_file_content_at_commit(file_path, "HEAD")
                new_content = self._get_current_file_content(file_path)
            
            if not old_content or not new_content:
                return changes
            
            # Parse dependencies from both versions
            old_deps = self._parse_dependency_content(file_path, old_content)
            new_deps = self._parse_dependency_content(file_path, new_content)
            
            ecosystem = self._get_ecosystem_from_file(file_path)
            
            # Compare dependencies
            all_deps = set(old_deps.keys()) | set(new_deps.keys())
            
            for dep_name in all_deps:
                old_version = old_deps.get(dep_name)
                new_version = new_deps.get(dep_name)
                
                if old_version is None and new_version is not None:
                    # Added dependency
                    change_type = 'added'
                elif old_version is not None and new_version is None:
                    # Removed dependency
                    change_type = 'removed'
                elif old_version != new_version:
                    # Updated dependency
                    change_type = 'updated'
                    if self._is_version_downgrade(old_version, new_version):
                        change_type = 'downgraded'
                else:
                    continue  # No change
                
                changes.append(DependencyChange(
                    name=dep_name,
                    old_version=old_version,
                    new_version=new_version,
                    change_type=change_type,
                    ecosystem=ecosystem,
                    breaking_changes=[],
                    security_impact=None
                ))
                
        except Exception as e:
            console.print(f"[yellow]Warning: Could not analyze {file_path}: {str(e)}[/yellow]")
        
        return changes
    
    def _get_file_content_at_commit(self, file_path: str, commit: str) -> Optional[str]:
        """Get file content at a specific commit"""
        try:
            result = run_git_command(['show', f'{commit}:{file_path}'], check=False)
            if result.returncode == 0:
                return result.stdout
        except Exception:
            pass
        return None
    
    def _get_current_file_content(self, file_path: str) -> Optional[str]:
        """Get current file content"""
        try:
            full_path = os.path.join(self.repo_root, file_path)
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None
    
    def _parse_dependency_content(self, file_path: str, content: str) -> Dict[str, str]:
        """Parse dependencies from file content"""
        deps = {}
        
        if 'package.json' in file_path:
            try:
                data = json.loads(content)
                deps.update(data.get('dependencies', {}))
                deps.update(data.get('devDependencies', {}))
            except:
                pass
        elif 'requirements' in file_path:
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    match = re.match(r'^([a-zA-Z0-9\-_\.]+)([><=!]+)?(.+)?', line)
                    if match:
                        name = match.group(1)
                        operator = match.group(2) or '=='
                        version = match.group(3) or 'latest'
                        deps[name] = f"{operator}{version}" if operator != '==' else version
        elif 'pyproject.toml' in file_path:
            # Basic TOML parsing for dependencies
            deps_match = re.search(r'\[tool\.poetry\.dependencies\](.*?)(?=\[|$)', content, re.DOTALL)
            if deps_match:
                deps_section = deps_match.group(1)
                for line in deps_section.split('\n'):
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        parts = line.split('=', 1)
                        if len(parts) == 2:
                            name = parts[0].strip()
                            version = parts[1].strip().strip('"\'')
                            if name != 'python':
                                deps[name] = version
        
        return deps
    
    def _get_ecosystem_from_file(self, file_path: str) -> str:
        """Determine ecosystem from file path"""
        if 'package.json' in file_path:
            return 'npm'
        elif 'requirements' in file_path:
            return 'pip'
        elif 'pyproject.toml' in file_path:
            return 'poetry'
        return 'unknown'
    
    def _is_version_downgrade(self, old_version: str, new_version: str) -> bool:
        """Check if version change is a downgrade (basic implementation)"""
        try:
            # Simple semantic version comparison
            old_parts = [int(x) for x in re.findall(r'\d+', old_version)]
            new_parts = [int(x) for x in re.findall(r'\d+', new_version)]
            
            # Pad with zeros to same length
            max_len = max(len(old_parts), len(new_parts))
            old_parts.extend([0] * (max_len - len(old_parts)))
            new_parts.extend([0] * (max_len - len(new_parts)))
            
            return new_parts < old_parts
        except:
            return False
    
    def generate_dependency_insights(self, changes: List[DependencyChange]) -> str:
        """Generate AI insights about dependency changes"""
        if not changes:
            return "No dependency changes detected."
        
        # Prepare context for AI
        context = "Dependency changes detected:\n\n"
        
        for change in changes:
            context += f"• {change.name} ({change.ecosystem}): "
            if change.change_type == 'added':
                context += f"Added version {change.new_version}\n"
            elif change.change_type == 'removed':
                context += f"Removed (was {change.old_version})\n"
            elif change.change_type == 'updated':
                context += f"Updated from {change.old_version} to {change.new_version}\n"
            elif change.change_type == 'downgraded':
                context += f"Downgraded from {change.old_version} to {change.new_version}\n"
        
        # Create AI prompt
        prompt = f"""Analyze these dependency changes and provide insights:

{context}

Focus on:
1. Security implications of changes
2. Potential breaking changes or compatibility issues
3. Performance impact
4. Recommended next steps
5. Any concerns or warnings

Provide a concise, actionable summary in 2-3 paragraphs."""

        try:
            provider = settings.get_provider()
            model = settings.get_model()
            
            if provider == 'openai':
                api_key = settings.get_openai_api_key()
                ai = OpenAIProvider(api_key)
            else:
                ai = OllamaProvider()
            
            insights = ai.generate_commit_message(prompt, model)
            return insights.strip()
            
        except Exception as e:
            return f"Could not generate AI insights: {str(e)}\n\nDependency changes summary:\n{context}"
    
    def check_for_updates(self, ecosystem: str = None) -> Dict[str, List[str]]:
        """Check for available dependency updates"""
        updates = {'npm': [], 'pip': [], 'poetry': []}
        
        if not ecosystem or ecosystem == 'npm':
            updates['npm'] = self._check_npm_updates()
        
        if not ecosystem or ecosystem in ['pip', 'poetry']:
            updates['pip'] = self._check_pip_updates()
            updates['poetry'] = self._check_poetry_updates()
        
        return updates
    
    def _check_npm_updates(self) -> List[str]:
        """Check for npm package updates"""
        updates = []
        try:
            # Look for package.json files
            for dep_file in self.dependency_files:
                if dep_file.ecosystem == 'npm':
                    file_dir = os.path.dirname(os.path.join(self.repo_root, dep_file.path))
                    try:
                        # Run npm outdated to check for updates
                        result = subprocess.run(
                            ['npm', 'outdated', '--json'],
                            cwd=file_dir,
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        if result.stdout:
                            outdated = json.loads(result.stdout)
                            for pkg, info in outdated.items():
                                current = info.get('current', '')
                                wanted = info.get('wanted', '')
                                latest = info.get('latest', '')
                                if current != wanted or current != latest:
                                    updates.append(f"{pkg}: {current} → {latest}")
                    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
                        continue
        except Exception:
            pass
        return updates
    
    def _check_pip_updates(self) -> List[str]:
        """Check for pip package updates"""
        updates = []
        try:
            # Get list of outdated packages
            result = subprocess.run(
                ['pip', 'list', '--outdated', '--format=json'],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0 and result.stdout:
                outdated = json.loads(result.stdout)
                for pkg in outdated:
                    name = pkg.get('name', '')
                    current = pkg.get('version', '')
                    latest = pkg.get('latest_version', '')
                    if current and latest:
                        updates.append(f"{name}: {current} → {latest}")
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            pass
        return updates
    
    def _check_poetry_updates(self) -> List[str]:
        """Check for poetry package updates"""
        updates = []
        try:
            for dep_file in self.dependency_files:
                if dep_file.ecosystem == 'poetry':
                    file_dir = os.path.dirname(os.path.join(self.repo_root, dep_file.path))
                    try:
                        # Run poetry show --outdated
                        result = subprocess.run(
                            ['poetry', 'show', '--outdated'],
                            cwd=file_dir,
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        if result.returncode == 0:
                            for line in result.stdout.split('\n'):
                                line = line.strip()
                                if line and not line.startswith('!'):
                                    # Parse poetry output format
                                    parts = line.split()
                                    if len(parts) >= 3:
                                        name = parts[0]
                                        current = parts[1]
                                        latest = parts[2]
                                        updates.append(f"{name}: {current} → {latest}")
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        continue
        except Exception:
            pass
        return updates
    
    def detect_breaking_changes(self, package_name: str, old_version: str, new_version: str, ecosystem: str) -> List[str]:
        """Detect potential breaking changes between versions"""
        breaking_changes = []
        
        try:
            # Simple semantic version analysis
            old_parts = [int(x) for x in re.findall(r'\d+', old_version)]
            new_parts = [int(x) for x in re.findall(r'\d+', new_version)]
            
            if len(old_parts) >= 1 and len(new_parts) >= 1:
                # Major version change
                if new_parts[0] > old_parts[0]:
                    breaking_changes.append(f"Major version update ({old_parts[0]}.x → {new_parts[0]}.x) - likely breaking changes")
                
                # Check for pre-release to stable
                if '0.0.' in old_version and not '0.0.' in new_version:
                    breaking_changes.append("Moving from pre-release to stable version")
                
                # Large minor version jumps might indicate breaking changes
                if len(old_parts) >= 2 and len(new_parts) >= 2:
                    if new_parts[0] == old_parts[0] and new_parts[1] - old_parts[1] > 5:
                        breaking_changes.append(f"Large minor version jump ({old_parts[1]} → {new_parts[1]}) - review changelog")
        
        except Exception:
            breaking_changes.append("Version format changed - manual review required")
        
        return breaking_changes
    
    def generate_update_plan(self, changes: List[DependencyChange]) -> str:
        """Generate AI-powered update plan with code adaptation suggestions"""
        if not changes:
            return "No dependency changes to plan for."
        
        # Analyze changes for potential issues
        high_risk_changes = []
        medium_risk_changes = []
        low_risk_changes = []
        
        for change in changes:
            breaking_changes = []
            if change.old_version and change.new_version:
                breaking_changes = self.detect_breaking_changes(
                    change.name, change.old_version, change.new_version, change.ecosystem
                )
            
            if breaking_changes:
                high_risk_changes.append((change, breaking_changes))
            elif change.change_type in ['updated', 'downgraded']:
                medium_risk_changes.append(change)
            else:
                low_risk_changes.append(change)
        
        # Create context for AI
        context = "Dependency update analysis:\n\n"
        
        if high_risk_changes:
            context += "HIGH RISK CHANGES (likely breaking):\n"
            for change, breaks in high_risk_changes:
                context += f"• {change.name}: {change.old_version} → {change.new_version}\n"
                for break_info in breaks:
                    context += f"  - {break_info}\n"
            context += "\n"
        
        if medium_risk_changes:
            context += "MEDIUM RISK CHANGES (version updates):\n"
            for change in medium_risk_changes:
                context += f"• {change.name}: {change.old_version} → {change.new_version}\n"
            context += "\n"
        
        if low_risk_changes:
            context += "LOW RISK CHANGES (additions/patches):\n"
            for change in low_risk_changes:
                if change.change_type == 'added':
                    context += f"• {change.name}: Added version {change.new_version}\n"
                else:
                    context += f"• {change.name}: {change.old_version} → {change.new_version}\n"
            context += "\n"
        
        # Create AI prompt for update planning
        prompt = f"""As a senior developer, create an update plan for these dependency changes:

{context}

Provide a practical update plan in this format:

## Update Strategy
(Overall approach - safe order, testing strategy)

## High Priority Actions
(Critical changes that need immediate attention)

## Code Changes Needed
(Specific code adaptations for breaking changes)

## Testing Plan
(What to test after updates)

## Rollback Plan
(How to revert if issues occur)

Keep it practical and actionable. Focus on real-world implications."""

        try:
            provider = settings.get_provider()
            model = settings.get_model()
            
            if provider == 'openai':
                api_key = settings.get_openai_api_key()
                ai = OpenAIProvider(api_key)
            else:
                ai = OllamaProvider()
            
            plan = ai.generate_commit_message(prompt, model)
            return plan.strip()
            
        except Exception as e:
            return f"Could not generate update plan: {str(e)}\n\nManual review required for:\n{context}"
    
    def suggest_code_adaptations(self, package_name: str, old_version: str, new_version: str, ecosystem: str) -> List[str]:
        """Suggest code adaptations for package updates"""
        adaptations = []
        
        # Common patterns for different ecosystems
        if ecosystem == 'npm':
            # Common Node.js breaking change patterns
            if 'express' in package_name.lower():
                adaptations.append("Check for middleware signature changes")
                adaptations.append("Review request/response object changes")
            elif 'react' in package_name.lower():
                adaptations.append("Check for prop type changes")
                adaptations.append("Review lifecycle method updates")
            elif 'webpack' in package_name.lower():
                adaptations.append("Update webpack configuration")
                adaptations.append("Check for loader changes")
        
        elif ecosystem in ['pip', 'poetry']:
            # Common Python breaking change patterns
            if 'django' in package_name.lower():
                adaptations.append("Check for model field changes")
                adaptations.append("Review URL configuration updates")
                adaptations.append("Check for middleware changes")
            elif 'flask' in package_name.lower():
                adaptations.append("Review decorator changes")
                adaptations.append("Check for request context updates")
            elif 'requests' in package_name.lower():
                adaptations.append("Check for session handling changes")
                adaptations.append("Review authentication method updates")
        
        # Add version-specific suggestions
        try:
            old_major = int(re.findall(r'\d+', old_version)[0])
            new_major = int(re.findall(r'\d+', new_version)[0])
            
            if new_major > old_major:
                adaptations.extend([
                    f"Review {package_name} migration guide for v{old_major} to v{new_major}",
                    "Check for deprecated API usage in your code",
                    "Update import statements if module structure changed"
                ])
        except (IndexError, ValueError):
            pass
        
        return adaptations
    
    def auto_update_dependencies(self, ecosystem: str = None, dry_run: bool = True) -> Dict[str, any]:
        """Automatically update dependencies with intelligent conflict resolution"""
        results = {
            'updated': [],
            'failed': [],
            'conflicts': [],
            'breaking_changes': []
        }
        
        try:
            if ecosystem == 'npm' or not ecosystem:
                results.update(self._auto_update_npm(dry_run))
            
            if ecosystem in ['pip', 'poetry'] or not ecosystem:
                results.update(self._auto_update_python(dry_run))
                
        except Exception as e:
            results['failed'].append(f"Auto-update failed: {str(e)}")
        
        return results
    
    def _auto_update_npm(self, dry_run: bool = True) -> Dict[str, List[str]]:
        """Auto-update npm dependencies"""
        results = {'updated': [], 'failed': [], 'conflicts': []}
        
        for dep_file in self.dependency_files:
            if dep_file.ecosystem == 'npm':
                file_dir = os.path.dirname(os.path.join(self.repo_root, dep_file.path))
                try:
                    if dry_run:
                        # Just check what would be updated
                        result = subprocess.run(
                            ['npm', 'outdated', '--json'],
                            cwd=file_dir,
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        if result.stdout:
                            outdated = json.loads(result.stdout)
                            for pkg in outdated.keys():
                                results['updated'].append(f"Would update {pkg}")
                    else:
                        # Actually update
                        result = subprocess.run(
                            ['npm', 'update'],
                            cwd=file_dir,
                            capture_output=True,
                            text=True,
                            timeout=120
                        )
                        if result.returncode == 0:
                            results['updated'].append(f"Updated npm packages in {dep_file.path}")
                        else:
                            results['failed'].append(f"Failed to update {dep_file.path}: {result.stderr}")
                
                except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
                    results['failed'].append(f"NPM update error: {str(e)}")
        
        return results
    
    def _auto_update_python(self, dry_run: bool = True) -> Dict[str, List[str]]:
        """Auto-update Python dependencies"""
        results = {'updated': [], 'failed': [], 'conflicts': []}
        
        # Handle pip requirements files
        for dep_file in self.dependency_files:
            if dep_file.ecosystem == 'pip':
                try:
                    if dry_run:
                        # Check what would be updated
                        result = subprocess.run(
                            ['pip', 'list', '--outdated', '--format=json'],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        if result.returncode == 0 and result.stdout:
                            outdated = json.loads(result.stdout)
                            for pkg in outdated:
                                results['updated'].append(f"Would update {pkg['name']}")
                    else:
                        # Note: Auto-updating pip requirements requires careful handling
                        results['failed'].append("Pip auto-update requires manual review of requirements.txt")
                
                except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
                    results['failed'].append(f"Pip update error: {str(e)}")
        
        # Handle poetry projects
        for dep_file in self.dependency_files:
            if dep_file.ecosystem == 'poetry':
                file_dir = os.path.dirname(os.path.join(self.repo_root, dep_file.path))
                try:
                    if dry_run:
                        # Check what would be updated
                        result = subprocess.run(
                            ['poetry', 'show', '--outdated'],
                            cwd=file_dir,
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        if result.returncode == 0:
                            lines = [l for l in result.stdout.split('\n') if l.strip()]
                            for line in lines:
                                if line.strip() and not line.startswith('!'):
                                    results['updated'].append(f"Would update via poetry: {line.strip()}")
                    else:
                        # Actually update with poetry
                        result = subprocess.run(
                            ['poetry', 'update'],
                            cwd=file_dir,
                            capture_output=True,
                            text=True,
                            timeout=120
                        )
                        if result.returncode == 0:
                            results['updated'].append(f"Updated poetry packages in {dep_file.path}")
                        else:
                            results['failed'].append(f"Failed to update {dep_file.path}: {result.stderr}")
                
                except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                    results['failed'].append(f"Poetry update error: {str(e)}")
        
        return results 
    
    def analyze_code_for_dependencies(self, commit_hash: str = None) -> List[DetectedDependency]:
        """Analyze code changes to detect new dependencies needed"""
        dependencies = []
        
        try:
            if commit_hash:
                # Get diff for specific commit
                diff = run_git_command(['show', '--no-color', commit_hash]).stdout
            else:
                # Get current staged/unstaged changes
                diff = run_git_command(['diff', 'HEAD']).stdout
                if not diff.strip():
                    # Try staged changes
                    diff = run_git_command(['diff', '--cached']).stdout
            
            if not diff.strip():
                return dependencies
            
            # Parse diff to find new imports/requires
            dependencies.extend(self._parse_diff_for_imports(diff))
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not analyze code for dependencies: {str(e)}[/yellow]")
        
        return dependencies
    
    def _parse_diff_for_imports(self, diff: str) -> List[DetectedDependency]:
        """Parse git diff to find new import statements"""
        dependencies = []
        
        lines = diff.split('\n')
        current_file = None
        line_number = 0
        
        for line in lines:
            # Track current file being processed
            if line.startswith('+++'):
                current_file = line[4:].strip()
                if current_file.startswith('b/'):
                    current_file = current_file[2:]
                line_number = 0
                continue
            
            # Track line numbers
            if line.startswith('@@'):
                match = re.search(r'\+(\d+)', line)
                if match:
                    line_number = int(match.group(1)) - 1
                continue
            
            # Only look at added lines
            if not line.startswith('+') or line.startswith('+++'):
                if not line.startswith('-'):
                    line_number += 1
                continue
            
            line_number += 1
            line_content = line[1:].strip()  # Remove the '+' prefix
            
            if not line_content or current_file is None:
                continue
            
            # Detect dependencies based on file type and import patterns
            detected = self._detect_dependency_from_line(line_content, current_file, line_number)
            if detected:
                dependencies.append(detected)
        
        return dependencies
    
    def _detect_dependency_from_line(self, line: str, file_path: str, line_number: int) -> Optional[DetectedDependency]:
        """Detect dependency from a single line of code"""
        
        # Python patterns
        if file_path.endswith('.py'):
            return self._detect_python_dependency(line, file_path, line_number)
        
        # JavaScript/TypeScript patterns
        elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
            return self._detect_js_dependency(line, file_path, line_number)
        
        return None
    
    def _detect_python_dependency(self, line: str, file_path: str, line_number: int) -> Optional[DetectedDependency]:
        """Detect Python import statements"""
        
        # Common Python import patterns
        import_patterns = [
            # import package
            r'^import\s+([a-zA-Z_][a-zA-Z0-9_\.]*)',
            # from package import something
            r'^from\s+([a-zA-Z_][a-zA-Z0-9_\.]*)\s+import',
        ]
        
        for pattern in import_patterns:
            match = re.match(pattern, line.strip())
            if match:
                import_name = match.group(1)
                
                # Skip standard library and relative imports
                if self._is_python_stdlib(import_name) or import_name.startswith('.'):
                    return None
                
                # Map import name to package name
                package_name = self._map_python_import_to_package(import_name)
                confidence = self._calculate_confidence(import_name, package_name, 'python')
                
                return DetectedDependency(
                    package_name=package_name,
                    import_statement=line.strip(),
                    file_path=file_path,
                    line_number=line_number,
                    ecosystem='pip',
                    confidence=confidence,
                    suggested_version=self._suggest_version(package_name, 'pip')
                )
        
        return None
    
    def _detect_js_dependency(self, line: str, file_path: str, line_number: int) -> Optional[DetectedDependency]:
        """Detect JavaScript/TypeScript import statements"""
        
        # Common JS/TS import patterns
        import_patterns = [
            # import something from 'package'
            r'import\s+.*\s+from\s+[\'"]([^\'\"]+)[\'"]',
            # import 'package'
            r'import\s+[\'"]([^\'\"]+)[\'"]',
            # const something = require('package')
            r'require\s*\(\s*[\'"]([^\'\"]+)[\'"]\s*\)',
        ]
        
        for pattern in import_patterns:
            match = re.search(pattern, line.strip())
            if match:
                import_name = match.group(1)
                
                # Skip relative imports
                if import_name.startswith('.') or import_name.startswith('/'):
                    return None
                
                # Map import name to package name
                package_name = self._map_js_import_to_package(import_name)
                confidence = self._calculate_confidence(import_name, package_name, 'javascript')
                
                return DetectedDependency(
                    package_name=package_name,
                    import_statement=line.strip(),
                    file_path=file_path,
                    line_number=line_number,
                    ecosystem='npm',
                    confidence=confidence,
                    suggested_version=self._suggest_version(package_name, 'npm')
                )
        
        return None
    
    def _is_python_stdlib(self, module_name: str) -> bool:
        """Check if a module is part of Python standard library"""
        # Common Python standard library modules
        stdlib_modules = {
            'os', 'sys', 'json', 're', 'datetime', 'time', 'math', 'random',
            'collections', 'itertools', 'functools', 'operator', 'pathlib',
            'typing', 'dataclasses', 'asyncio', 'threading', 'multiprocessing',
            'subprocess', 'urllib', 'http', 'email', 'xml', 'html', 'csv',
            'sqlite3', 'logging', 'unittest', 'argparse', 'configparser',
            'io', 'base64', 'hashlib', 'hmac', 'secrets', 'uuid', 'pickle',
            'copy', 'pprint', 'weakref', 'gc', 'inspect', 'warnings',
            'contextlib', 'abc', 'numbers', 'fractions', 'decimal',
            'statistics', 'enum', 'array', 'struct', 'codecs', 'string',
            'textwrap', 'locale', 'calendar', 'tempfile', 'glob', 'fnmatch',
            'shutil', 'platform', 'socket', 'ssl', 'select', 'signal',
            'mmap', 'ctypes', 'winreg', 'posix', 'pwd', 'grp', 'crypt',
            'readline', 'rlcompleter', 'zipfile', 'tarfile', 'gzip',
            'bz2', 'lzma', 'zlib', 'binascii', 'quopri', 'uu'
        }
        
        # Check if it's a direct stdlib module or submodule
        top_level = module_name.split('.')[0]
        return top_level in stdlib_modules
    
    def _map_python_import_to_package(self, import_name: str) -> str:
        """Map Python import name to package name"""
        
        # Common mappings where import name != package name
        import_to_package = {
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'skimage': 'scikit-image',
            'sklearn': 'scikit-learn',
            'yaml': 'PyYAML',
            'dotenv': 'python-dotenv',
            'jwt': 'PyJWT',
            'bs4': 'beautifulsoup4',
            'serial': 'pyserial',
            'magic': 'python-magic',
            'dateutil': 'python-dateutil',
            'slugify': 'python-slugify',
            'crontab': 'python-crontab',
            'OpenSSL': 'pyOpenSSL',
            'psycopg2': 'psycopg2-binary',
            'MySQLdb': 'mysqlclient',
            'cx_Oracle': 'cx-Oracle',
        }
        
        # Check for direct mapping
        if import_name in import_to_package:
            return import_to_package[import_name]
        
        # For submodules, try the top-level package
        top_level = import_name.split('.')[0]
        if top_level in import_to_package:
            return import_to_package[top_level]
        
        # Default: assume import name is package name
        return import_name
    
    def _map_js_import_to_package(self, import_name: str) -> str:
        """Map JavaScript import name to package name"""
        
        # Common mappings and scoped packages
        import_to_package = {
            'lodash': 'lodash',
            '_': 'lodash',
            'moment': 'moment',
            'axios': 'axios',
            'react': 'react',
            'vue': 'vue',
            'angular': '@angular/core',
            'express': 'express',
            'mongoose': 'mongoose',
            'sequelize': 'sequelize',
            'socket.io': 'socket.io',
            'ws': 'ws',
            'bcrypt': 'bcrypt',
            'jwt': 'jsonwebtoken',
            'multer': 'multer',
            'cors': 'cors',
            'helmet': 'helmet',
            'dotenv': 'dotenv',
            'nodemon': 'nodemon',
            'webpack': 'webpack',
            'babel': '@babel/core',
            'eslint': 'eslint',
            'prettier': 'prettier',
            'jest': 'jest',
            'mocha': 'mocha',
            'chai': 'chai',
            'sinon': 'sinon',
        }
        
        # Handle scoped packages (e.g., @babel/core)
        if import_name.startswith('@'):
            return import_name
        
        # Check for direct mapping
        if import_name in import_to_package:
            return import_to_package[import_name]
        
        # For submodules, try the top-level package
        top_level = import_name.split('/')[0]
        if top_level in import_to_package:
            return import_to_package[top_level]
        
        # Default: assume import name is package name
        return import_name
    
    def _calculate_confidence(self, import_name: str, package_name: str, language: str) -> float:
        """Calculate confidence level for dependency detection"""
        confidence = 0.7  # Base confidence
        
        # Higher confidence for exact matches
        if import_name == package_name:
            confidence += 0.2
        
        # Higher confidence for common packages
        common_packages = {
            'python': ['requests', 'numpy', 'pandas', 'django', 'flask', 'fastapi'],
            'javascript': ['react', 'vue', 'angular', 'express', 'lodash', 'axios']
        }
        
        if package_name in common_packages.get(language, []):
            confidence += 0.1
        
        # Lower confidence for single character or very short names
        if len(package_name) <= 2:
            confidence -= 0.3
        
        return min(1.0, max(0.1, confidence))
    
    def _suggest_version(self, package_name: str, ecosystem: str) -> Optional[str]:
        """Suggest a version for the package"""
        try:
            if ecosystem == 'npm':
                # Query npm registry for latest version
                response = requests.get(f"https://registry.npmjs.org/{package_name}/latest", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    return f"^{data.get('version', 'latest')}"
            
            elif ecosystem == 'pip':
                # Query PyPI for latest version
                response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    return f">={data['info']['version']}"
        
        except (requests.RequestException, KeyError, json.JSONDecodeError):
            pass
        
        return None
    
    def auto_add_dependencies(self, dependencies: List[DetectedDependency], dry_run: bool = True) -> Dict[str, List[str]]:
        """Automatically add detected dependencies to dependency files"""
        results = {
            'added': [],
            'failed': [],
            'skipped': [],
            'files_updated': []
        }
        
        if not dependencies:
            return results
        
        # Group dependencies by ecosystem and file
        ecosystems = {}
        for dep in dependencies:
            if dep.ecosystem not in ecosystems:
                ecosystems[dep.ecosystem] = []
            ecosystems[dep.ecosystem].append(dep)
        
        # Process each ecosystem
        for ecosystem, deps in ecosystems.items():
            try:
                if ecosystem == 'npm':
                    result = self._add_npm_dependencies(deps, dry_run)
                elif ecosystem in ['pip', 'poetry']:
                    result = self._add_python_dependencies(deps, dry_run)
                else:
                    continue
                
                # Merge results
                for key in results:
                    if key in result:
                        results[key].extend(result[key])
            
            except Exception as e:
                results['failed'].append(f"Failed to add {ecosystem} dependencies: {str(e)}")
        
        return results
    
    def _add_npm_dependencies(self, dependencies: List[DetectedDependency], dry_run: bool = True) -> Dict[str, List[str]]:
        """Add npm dependencies to package.json"""
        results = {'added': [], 'failed': [], 'skipped': [], 'files_updated': []}
        
        # Find package.json files
        package_json_files = []
        for dep_file in self.dependency_files:
            if dep_file.ecosystem == 'npm':
                package_json_files.append(dep_file.path)
        
        if not package_json_files:
            results['failed'].append("No package.json found")
            return results
        
        # Use the first package.json (or closest to root)
        package_json_path = min(package_json_files, key=lambda x: len(x.split('/')))
        full_path = os.path.join(self.repo_root, package_json_path)
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                package_data = json.load(f)
            
            # Add dependencies
            if 'dependencies' not in package_data:
                package_data['dependencies'] = {}
            
            for dep in dependencies:
                if dep.confidence < 0.5:  # Skip low confidence dependencies
                    results['skipped'].append(f"{dep.package_name} (low confidence: {dep.confidence:.2f})")
                    continue
                
                if dep.package_name in package_data['dependencies']:
                    results['skipped'].append(f"{dep.package_name} (already exists)")
                    continue
                
                version = dep.suggested_version or "^latest"
                
                if dry_run:
                    results['added'].append(f"Would add {dep.package_name}@{version}")
                else:
                    package_data['dependencies'][dep.package_name] = version
                    results['added'].append(f"Added {dep.package_name}@{version}")
            
            # Write back to file if not dry run
            if not dry_run and results['added']:
                with open(full_path, 'w', encoding='utf-8') as f:
                    json.dump(package_data, f, indent=2, ensure_ascii=False)
                    f.write('\n')  # Add trailing newline
                results['files_updated'].append(package_json_path)
        
        except Exception as e:
            results['failed'].append(f"Failed to update package.json: {str(e)}")
        
        return results
    
    def _add_python_dependencies(self, dependencies: List[DetectedDependency], dry_run: bool = True) -> Dict[str, List[str]]:
        """Add Python dependencies to requirements.txt or pyproject.toml"""
        results = {'added': [], 'failed': [], 'skipped': [], 'files_updated': []}
        
        # Find appropriate dependency file
        target_file = None
        target_type = None
        
        # Prefer pyproject.toml if it exists
        for dep_file in self.dependency_files:
            if dep_file.ecosystem == 'poetry':
                target_file = dep_file.path
                target_type = 'poetry'
                break
        
        # Fall back to requirements.txt
        if not target_file:
            for dep_file in self.dependency_files:
                if dep_file.ecosystem == 'pip':
                    target_file = dep_file.path
                    target_type = 'pip'
                    break
        
        # Create requirements.txt if no dependency file exists
        if not target_file:
            target_file = 'requirements.txt'
            target_type = 'pip'
        
        full_path = os.path.join(self.repo_root, target_file)
        
        try:
            if target_type == 'poetry':
                results.update(self._add_to_pyproject_toml(dependencies, full_path, dry_run))
            else:
                results.update(self._add_to_requirements_txt(dependencies, full_path, dry_run))
            
            if results['added'] and not dry_run:
                results['files_updated'].append(target_file)
        
        except Exception as e:
            results['failed'].append(f"Failed to update {target_file}: {str(e)}")
        
        return results
    
    def _add_to_requirements_txt(self, dependencies: List[DetectedDependency], file_path: str, dry_run: bool) -> Dict[str, List[str]]:
        """Add dependencies to requirements.txt"""
        results = {'added': [], 'failed': [], 'skipped': []}
        
        # Read existing requirements
        existing_packages = set()
        lines = []
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    package_name = re.match(r'^([a-zA-Z0-9\-_\.]+)', line)
                    if package_name:
                        existing_packages.add(package_name.group(1).lower())
        
        # Add new dependencies
        new_lines = []
        for dep in dependencies:
            if dep.confidence < 0.5:
                results['skipped'].append(f"{dep.package_name} (low confidence: {dep.confidence:.2f})")
                continue
            
            if dep.package_name.lower() in existing_packages:
                results['skipped'].append(f"{dep.package_name} (already exists)")
                continue
            
            version = dep.suggested_version or ""
            new_line = f"{dep.package_name}{version}\n"
            
            if dry_run:
                results['added'].append(f"Would add {dep.package_name}{version}")
            else:
                new_lines.append(new_line)
                results['added'].append(f"Added {dep.package_name}{version}")
        
        # Write back to file
        if not dry_run and new_lines:
            with open(file_path, 'a', encoding='utf-8') as f:
                if lines and not lines[-1].endswith('\n'):
                    f.write('\n')  # Ensure newline before adding
                f.writelines(new_lines)
        
        return results
    
    def _add_to_pyproject_toml(self, dependencies: List[DetectedDependency], file_path: str, dry_run: bool) -> Dict[str, List[str]]:
        """Add dependencies to pyproject.toml (basic implementation)"""
        results = {'added': [], 'failed': [], 'skipped': []}
        
        # Note: This is a basic implementation without a full TOML parser
        # In production, you'd want to use a proper TOML library like `toml` or `tomli`
        
        for dep in dependencies:
            if dep.confidence < 0.5:
                results['skipped'].append(f"{dep.package_name} (low confidence: {dep.confidence:.2f})")
                continue
            
            if dry_run:
                results['added'].append(f"Would add {dep.package_name} to pyproject.toml")
            else:
                # For now, suggest manual addition
                results['failed'].append(f"Auto-add to pyproject.toml not fully implemented. Manually add: {dep.package_name}")
        
        return results 