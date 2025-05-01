#!/usr/bin/env python3
"""
Cross-platform script to run the FastAPI backend and React frontend.
Works on Windows, macOS, and Linux without any additional dependencies.
"""

import os
import sys
import time
import signal
import webbrowser
import subprocess
import platform
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Configuration
PORT_BACKEND = os.getenv("PORT_BACKEND")
PORT_FRONTEND = os.getenv("PORT_FRONTEND")
PROCESSES = []


def is_windows():
    """Check if the current platform is Windows"""
    return platform.system() == "Windows"


def find_process_by_port(port):
    """Find process using a specific port"""
    if is_windows():
        try:
            # Windows - use netstat
            output = subprocess.check_output(
                f"netstat -ano | findstr :{port}", shell=True
            ).decode()
            if output:
                for line in output.splitlines():
                    if f":{port}" in line and "LISTENING" in line:
                        parts = line.strip().split()
                        return int(parts[-1])
        except subprocess.CalledProcessError:
            pass
    else:
        try:
            # Unix-like - use lsof
            output = (
                subprocess.check_output(f"lsof -i :{port} -t", shell=True)
                .decode()
                .strip()
            )
            if output:
                return int(output)
        except subprocess.CalledProcessError:
            pass
    return None


def kill_process(pid):
    """Kill a process by its PID"""
    if pid:
        try:
            if is_windows():
                subprocess.run(f"taskkill /F /PID {pid}", shell=True, check=False)
            else:
                os.kill(pid, signal.SIGKILL)
            print(f"[INFO] Killed process {pid}")
            return True
        except (subprocess.SubprocessError, OSError) as e:
            print(f"[WARNING] Failed to kill process {pid}: {e}")
    return False


def cleanup_port(port):
    """Clean up a specific port by killing any process using it"""
    pid = find_process_by_port(port)
    if pid:
        print(f"[INFO] Port {port} is in use by process {pid}, attempting to free...")
        kill_process(pid)
        # Wait briefly to ensure the port is released
        time.sleep(1)
        return True
    return False


def cleanup():
    """Clean up all running processes and ports"""
    print("\n[INFO] Shutting down all processes...")

    # First try to terminate processes gracefully
    for proc in PROCESSES:
        if proc and proc.poll() is None:  # Check if process is still running
            try:
                if is_windows():
                    proc.terminate()
                else:
                    proc.send_signal(signal.SIGTERM)
                print(f"[INFO] Terminating process: {proc.pid}")
            except Exception as e:
                print(f"[WARNING] Error terminating process {proc.pid}: {e}")

    # Give processes time to shut down gracefully
    time.sleep(2)

    # Force kill any remaining processes
    for proc in PROCESSES:
        if proc and proc.poll() is None:  # Check if process is still running
            try:
                proc.kill()
                print(f"[INFO] Force killed process: {proc.pid}")
            except Exception as e:
                print(f"[WARNING] Error killing process {proc.pid}: {e}")

    # Clean up ports as a last resort
    cleanup_port(PORT_BACKEND)
    cleanup_port(PORT_FRONTEND)

    print("[INFO] All processes terminated, ports released.")


def signal_handler(sig, frame):
    """Handle interrupt signals (Ctrl+C)"""
    print("\n[INFO] Received interrupt signal. Shutting down...")
    cleanup()
    sys.exit(0)


def run_command(cmd, cwd=None, shell=False):
    """Run a command in a subprocess with appropriate platform considerations"""
    if is_windows() and not shell:
        # Windows needs shell=True unless it's a list of args
        if isinstance(cmd, str):
            return subprocess.Popen(cmd, cwd=cwd, shell=True)
        else:
            return subprocess.Popen(cmd, cwd=cwd)
    else:
        return subprocess.Popen(cmd, cwd=cwd, shell=shell)


def main():
    """Main function to start services"""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Set environment variable
    os.environ["REPORT_TOOL_TRACES"] = "1"

    print("[INFO] Starting services. Press Ctrl+C to stop all processes.")

    # Clean up ports if they're already in use
    cleanup_port(PORT_BACKEND)
    cleanup_port(PORT_FRONTEND)

    try:
        # Start FastAPI backend
        if is_windows():
            backend_proc = run_command(
                f"python -m fastapi dev app.py --port {PORT_BACKEND}"
            )
        else:
            backend_proc = run_command(
                f"fastapi dev app.py --port {PORT_BACKEND}", shell=True
            )

        PROCESSES.append(backend_proc)
        print(f"[INFO] Started FastAPI backend with process ID: {backend_proc.pid}")

        # Start React frontend
        chat_app_dir = Path("chat-app").absolute()
        if not chat_app_dir.exists():
            print(f"[ERROR] Directory not found: {chat_app_dir}")
            cleanup()
            sys.exit(1)

        if is_windows():
            frontend_proc = run_command(
                f"$env:PORT={PORT_FRONTEND}; npm run dev",
                cwd=str(chat_app_dir),
                shell=True,
            )
        else:
            frontend_proc = subprocess.Popen(
                f"PORT={PORT_FRONTEND} npm run dev",
                cwd=str(chat_app_dir),
                shell=False,
            )

        PROCESSES.append(frontend_proc)
        print(f"[INFO] Started React frontend with process ID: {frontend_proc.pid}")

        # Give services a moment to start
        time.sleep(2)

        # Open browser
        print("[INFO] Opening browser to http://localhost:3000")
        try:
            webbrowser.open(f"http://localhost:{PORT_FRONTEND}")
        except Exception as e:
            print(f"[WARNING] Could not open browser: {e}")
            print(f"[INFO] Please manually open: http://localhost:{PORT_FRONTEND}")

        # Keep the script running until Ctrl+C
        print("[INFO] Services are running. Press Ctrl+C to stop.")
        while all(proc.poll() is None for proc in PROCESSES if proc):
            time.sleep(1)

        # If we get here, one of the processes has ended
        for proc in PROCESSES:
            if proc and proc.poll() is not None:
                print(
                    f"[WARNING] Process {proc.pid} exited with code {proc.returncode}"
                )

        # Clean up remaining processes
        cleanup()

    except Exception as e:
        print(f"[ERROR] {e}")
        cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
