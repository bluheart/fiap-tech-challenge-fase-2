#!/usr/bin/env python3
"""
Environment Validation Script for Python Projects
Checks Python version, UV package manager, environment variables, and system requirements
"""

import sys
import os
import subprocess
import platform
import shutil
import json
from pathlib import Path
from typing import Optional
import warnings

# Suppress warnings during validation
warnings.filterwarnings("ignore")


class EnvironmentValidator:
    """Validates the project environment setup"""

    def __init__(self, project_root: Optional[str] = None):
        # If project_root not provided, find it (script is in scripts/ directory)
        if project_root is None:
            # Get the directory where this script is located
            script_dir = Path(__file__).parent.resolve()
            # Project root is one level up from scripts/
            self.project_root = script_dir.parent.resolve()
        else:
            self.project_root = Path(project_root).resolve()

        self.errors = []
        self.warnings = []
        self.successes = []
        self.required_python_version = (3, 8)  # Minimum required version
        self.recommended_python_version = (3, 11)  # Recommended version

    def run_all_checks(self) -> bool:
        """Run all validation checks"""
        print("\n" + "=" * 60)
        print("🔍 ENVIRONMENT VALIDATION STARTED")
        print(f"📁 Project Root: {self.project_root}")
        print("=" * 60)

        # Run all checks
        self.check_python_version()
        self.check_os_compatibility()
        self.check_uv_and_package_manager()
        self.check_uv_dependencies()
        self.check_project_structure()
        self.check_environment_variables()
        self.check_git()
        self.check_virtual_environment()
        self.check_system_requirements()
        self.check_config_files()

        # Print results
        self.print_results()

        # Return if validation passed
        return len(self.errors) == 0

    def check_python_version(self):
        """Check Python version meets requirements"""
        version = sys.version_info
        current_version = (version.major, version.minor)

        if current_version < self.required_python_version:
            self.errors.append(
                f"Python version {version.major}.{version.minor}.{version.micro} "
                f"is below minimum required {self.required_python_version[0]}.{self.required_python_version[1]}"
            )
        elif current_version < self.recommended_python_version:
            self.warnings.append(
                f"Python version {version.major}.{version.minor}.{version.micro} "
                f"is below recommended {self.recommended_python_version[0]}.{self.recommended_python_version[1]}"
            )
        else:
            self.successes.append(
                f"Python version {version.major}.{version.minor}.{version.micro} ✓"
            )

    def check_os_compatibility(self):
        """Check operating system"""
        system = platform.system()
        version = platform.version()

        if system in ["Linux", "Darwin"]:  # Linux or macOS
            self.successes.append(f"Operating System: {system} {version} ✓")
        elif system == "Windows":
            self.warnings.append(
                f"Operating System: {system} {version} - Limited support may be available"
            )
        else:
            self.warnings.append(f"Unrecognized operating system: {system}")

    def check_uv_and_package_manager(self):
        """Check UV package manager installation"""
        # Check if UV is installed
        uv_path = shutil.which("uv")
        if uv_path:
            try:
                # Get UV version
                result = subprocess.run(
                    ["uv", "--version"], capture_output=True, text=True, check=True
                )
                version = result.stdout.strip()
                self.successes.append(f"UV package manager installed: {version} ✓")
                self.successes.append(f"UV path: {uv_path} ✓")

                # Check if UV tool is up to date
                try:
                    result = subprocess.run(
                        ["uv", "tool", "list"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    self.successes.append("UV tools available ✓")
                except subprocess.CalledProcessError:
                    self.warnings.append("Could not list UV tools")

            except subprocess.CalledProcessError:
                self.errors.append("UV installed but could not get version")
        else:
            self.errors.append(
                "UV package manager not found in PATH. Install with: pip install uv"
            )

    def check_uv_dependencies(self):
        """Check if dependencies are installed using UV"""
        # Check if pyproject.toml exists (UV uses this)
        pyproject_path = self.project_root / "pyproject.toml"
        if not pyproject_path.exists():
            self.warnings.append(
                "pyproject.toml not found (UV uses this for dependencies)"
            )
            return

        try:
            # Check if UV environment exists
            uv_venv = self.project_root / ".venv"
            if uv_venv.exists():
                self.successes.append("UV virtual environment (.venv) found ✓")
            else:
                self.warnings.append(
                    "UV virtual environment (.venv) not found. Run: uv venv"
                )

            # Try to sync dependencies
            result = subprocess.run(
                ["uv", "sync", "--dry-run"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                self.successes.append("UV dependencies check passed ✓")

                # Parse dependencies from the output
                if "No changes" in result.stdout:
                    self.successes.append("All dependencies are up to date ✓")
                else:
                    # Try to parse what needs to be installed
                    lines = result.stdout.split("\n")
                    for line in lines:
                        if "Added" in line or "Updated" in line:
                            self.warnings.append(
                                f"Dependency changes detected: {line.strip()}"
                            )
            else:
                self.errors.append(f"UV sync failed: {result.stderr.strip()}")

        except subprocess.CalledProcessError as e:
            self.errors.append(f"UV check failed: {e}")
        except Exception as e:
            self.errors.append(f"Error checking UV dependencies: {e}")

        # Check if lock file exists
        lock_file = self.project_root / "uv.lock"
        if lock_file.exists():
            self.successes.append("UV lock file (uv.lock) found ✓")
        else:
            self.warnings.append("UV lock file not found. Run: uv lock")

    def check_project_structure(self):
        """Check project directory structure"""
        # Check if we're in the right directory
        if not self.project_root.exists():
            self.errors.append(f"Project root does not exist: {self.project_root}")
            return

        # Check for common project directories relative to root
        project_dirs = ["src", "tests", "scripts", "data", "docs", "examples"]
        for dir_name in project_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                self.successes.append(f"Directory {dir_name}/ exists ✓")

        # Check if scripts directory contains this script
        scripts_dir = self.project_root / "scripts"
        if scripts_dir.exists() and scripts_dir.is_dir():
            self.successes.append("scripts/ directory found ✓")
            # Check for common scripts
            script_files = ["validate_env.py", "setup.sh", "run.py"]
            for script in script_files:
                if (scripts_dir / script).exists():
                    self.successes.append(f"  - {script} found ✓")

    def check_environment_variables(self):
        """Check required environment variables"""
        required_vars = ["PYTHONPATH"]

        optional_vars = [
            "OPENAI_API_KEY",
            "DATABASE_URL",
            "AWS_ACCESS_KEY_ID",
            "SECRET_KEY",
            "PROJECT_ROOT",
        ]

        for var in required_vars:
            value = os.environ.get(var)
            if value:
                self.successes.append(f"Environment variable {var} set ✓")
            else:
                self.warnings.append(f"Environment variable {var} not set (optional)")

        # Set PROJECT_ROOT if not set
        if not os.environ.get("PROJECT_ROOT"):
            os.environ["PROJECT_ROOT"] = str(self.project_root)
            self.successes.append(f"PROJECT_ROOT set to {self.project_root} ✓")

        # Check for common API keys
        sensitive_vars = []
        for var in optional_vars:
            if os.environ.get(var):
                sensitive_vars.append(var)

        if sensitive_vars:
            self.successes.append(f"API keys configured: {', '.join(sensitive_vars)} ✓")

    def check_git(self):
        """Check git installation and repository status"""
        # Check if git is installed
        if not shutil.which("git"):
            self.warnings.append("git not found in PATH (optional)")
            return

        try:
            # Check if directory is a git repository
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                self.successes.append("Git repository initialized ✓")

                # Check git status
                status = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                )

                if status.stdout:
                    self.warnings.append("Uncommitted changes in repository")
                else:
                    self.successes.append("Git working directory clean ✓")
            else:
                self.warnings.append("Not a git repository (optional)")
        except Exception:
            self.warnings.append("Git check failed (optional)")

    def check_virtual_environment(self):
        """Check if running in a virtual environment"""
        in_venv = sys.prefix != sys.base_prefix
        if in_venv:
            self.successes.append(f"Virtual environment active: {sys.prefix} ✓")

            # Check if it's a UV managed venv
            uv_venv = self.project_root / ".venv"
            if uv_venv.exists():
                self.successes.append("UV virtual environment detected ✓")

            # Check if dependencies are installed
            try:
                result = subprocess.run(
                    ["uv", "pip", "list", "--format=json"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    packages = json.loads(result.stdout)
                    self.successes.append(
                        f"Virtual environment has {len(packages)} packages ✓"
                    )
            except Exception:
                pass
        else:
            self.warnings.append(
                "Running in system Python (recommend using UV virtual environment)"
            )
            self.warnings.append(
                "Create with: uv venv && source .venv/bin/activate (Linux/Mac) or .venv\\Scripts\\activate (Windows)"
            )

    def check_system_requirements(self):
        """Check system requirements"""
        # Check disk space
        try:
            total, used, free = shutil.disk_usage(self.project_root)
            free_gb = free / (1024**3)
            if free_gb < 1:
                self.warnings.append(f"Low disk space: {free_gb:.2f} GB free")
            elif free_gb < 5:
                self.warnings.append(
                    f"Limited disk space: {free_gb:.2f} GB free (recommend >5GB)"
                )
            else:
                self.successes.append(f"Disk space available: {free_gb:.2f} GB ✓")
        except Exception:
            pass

        # Check memory (if psutil available)
        try:
            import psutil

            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            if memory_gb < 8:
                self.warnings.append(
                    f"Limited system memory: {memory_gb:.1f} GB (recommend >8GB)"
                )
            else:
                self.successes.append(f"System memory: {memory_gb:.1f} GB ✓")
        except ImportError:
            pass  # psutil is optional

    def check_config_files(self):
        """Check configuration files"""
        config_files = [
            (".env", "Environment configuration"),
            ("pyproject.toml", "Python project configuration (UV)"),
            ("uv.lock", "UV lock file"),
            ("config.yaml", "YAML configuration"),
            ("config.yml", "YAML configuration"),
            ("settings.json", "JSON settings"),
            (".gitignore", "Git ignore file"),
        ]

        for filename, description in config_files:
            file_path = self.project_root / filename
            if file_path.exists():
                self.successes.append(f"{description} file {filename} found ✓")
            else:
                # Check if there's a template
                template_path = self.project_root / f"{filename}.template"
                if template_path.exists():
                    self.successes.append(f"Template for {filename} found ✓")
                    self.warnings.append(
                        f"Configuration file {filename} missing (template available)"
                    )
                else:
                    # Only warn for important files
                    if filename in ["pyproject.toml", "uv.lock"]:
                        self.warnings.append(f"Important file {filename} not found")

    def print_results(self):
        """Print validation results"""
        print("\n" + "=" * 60)
        print("📊 VALIDATION RESULTS")
        print("=" * 60)

        if self.successes:
            print(f"\n✅ SUCCESSES ({len(self.successes)}):")
            for success in self.successes:
                print(f"  ✓ {success}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  ✗ {error}")

        print("\n" + "=" * 60)

        if self.errors:
            print("❌ VALIDATION FAILED - Please fix errors above")
            print("=" * 60)
        elif self.warnings:
            print("⚠️  VALIDATION PASSED WITH WARNINGS")
            print("=" * 60)
        else:
            print("✅ VALIDATION PASSED - Environment is ready!")
            print("=" * 60)


def setup_environment():
    """Helper function to set up common environment variables"""
    # Get the script directory
    script_dir = Path(__file__).parent.resolve()
    # Project root is one level up from scripts/
    project_root = script_dir.parent.resolve()

    # Add project root to Python path if not already
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Also add the scripts directory to path
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    # Create necessary directories if they don't exist
    (project_root / "data").mkdir(exist_ok=True)
    (project_root / "logs").mkdir(exist_ok=True)
    (project_root / "output").mkdir(exist_ok=True)
    (project_root / "scripts").mkdir(exist_ok=True)

    # Set project root environment variable
    os.environ["PROJECT_ROOT"] = str(project_root)

    print("✅ Environment setup complete")
    print(f"   Project root: {project_root}")
    print(f"   Scripts directory: {script_dir}")
    print(f"   Python path: {sys.path[0]}")

    return project_root


def main():
    """Main entry point"""
    # Setup environment
    project_root = setup_environment()

    # Run validation
    validator = EnvironmentValidator(project_root)  # type: ignore
    success = validator.run_all_checks()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
