#!/usr/bin/env python3
"""
Cross-platform setup script that works on Windows, macOS, and Linux.
Installs required dependencies for both backend and frontend.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def is_windows():
    """Check if the current platform is Windows"""
    return platform.system() == "Windows"

def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result"""
    print(f"Running: {cmd}")
    try:
        if isinstance(cmd, list):
            result = subprocess.run(cmd, cwd=cwd, check=check)
        else:
            result = subprocess.run(cmd, cwd=cwd, shell=True, check=check)
        return result.returncode == 0
    except subprocess.SubprocessError as e:
        print(f"Error running command: {e}")
        return False

def main():
    """Main setup function"""
    print("\n=== Setting up your application ===\n")
    
    # Install backend dependencies
    print("\n[1/2] Installing backend dependencies...")
    if not run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]):
        print("Failed to install backend requirements")
        return False
    
    # Set up frontend
    print("\n[2/2] Installing and building frontend...")
    chat_app_dir = Path('chat-app').absolute()
    if not chat_app_dir.exists():
        print(f"Error: Directory not found: {chat_app_dir}")
        return False
    
    # Check if package.json exists
    if not (chat_app_dir / "package.json").exists():
        print(f"Error: package.json not found in {chat_app_dir}")
        return False

    # Install npm dependencies 
    print("Installing npm dependencies...")
    if not run_command("npm install", cwd=str(chat_app_dir)):
        print("Failed to install npm dependencies")
        return False
    
    # Build frontend if needed (if there's a build script in package.json)
    if (chat_app_dir / "package.json").read_text().find('"build"') >= 0:
        print("Building frontend...")
        if not run_command("npm run build", cwd=str(chat_app_dir)):
            print("Warning: Frontend build may have issues, but continuing...")
    
    print("\n✅ Setup complete! You can now run the application with: python run.py")
    return True

if __name__ == "__main__":
    if not main():
        print("\n❌ Setup failed. Please check the errors above and try again.")
        sys.exit(1)
    sys.exit(0)