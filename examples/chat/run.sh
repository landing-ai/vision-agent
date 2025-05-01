#!/bin/bash

export REPORT_TOOL_TRACES=1
export $(cat .env | xargs)

# Store the process IDs for later cleanup
PIDS=()

# Function to clean up all background processes
cleanup() {
  echo -e "\n[INFO] Shutting down all processes..."
  
  # Kill concurrently and its child processes first
  if [ -n "$CONCURRENTLY_PID" ]; then
    kill -TERM "$CONCURRENTLY_PID" 2>/dev/null
  fi
  
  # Kill all tracked processes with increasing force if needed
  for pid in "${PIDS[@]}"; do
    if ps -p "$pid" > /dev/null 2>&1; then
      kill -TERM "$pid" 2>/dev/null
      
      # Give process a moment to terminate gracefully
      sleep 0.5
      
      # If still running, force kill
      if ps -p "$pid" > /dev/null 2>&1; then
        kill -9 "$pid" 2>/dev/null
      fi
    fi
  done
  
  # Force kill any processes still using our ports
  echo "[INFO] Releasing ports..."
  for port in $PORT_FRONTEND $PORT_BACKEND; do
    pid=$(lsof -ti:$port 2>/dev/null)
    if [ -n "$pid" ]; then
      echo "[INFO] Killing process $pid using port $port"
      kill -9 $pid 2>/dev/null
    fi
  done
  
  echo "[INFO] All processes terminated, ports released."
  exit 0
}

# Make sure the script exits even if a subcommand hangs
kill_after_timeout() {
  sleep 5
  echo "[WARNING] Force killing remaining processes after timeout"
  pkill -P $$ 2>/dev/null
  for port in $PORT_FRONTEND $PORT_BACKEND; do
    lsof -ti:$port | xargs -r kill -9 2>/dev/null
  done
  exit 1
}

# Register the cleanup function for different signals
# Immediate propagation with SIG_IGN first followed by cleanup
trap 'trap " " SIGTERM SIGINT; cleanup' SIGINT SIGTERM
trap cleanup EXIT

echo "[INFO] Starting services. Press Ctrl+C once to stop all processes."

# Start services but capture their PIDs
concurrently "fastapi dev app.py --port $PORT_BACKEND" "cd chat-app && PORT=$PORT_FRONTEND npm run dev" &
CONCURRENTLY_PID=$!
PIDS+=($CONCURRENTLY_PID)

# Find child processes and add them to our list
sleep 2
CHILD_PIDS=$(pgrep -P $CONCURRENTLY_PID 2>/dev/null)
if [ -n "$CHILD_PIDS" ]; then
  for pid in $CHILD_PIDS; do
    PIDS+=($pid)
  done
fi

# Try to open browser
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open http://localhost:$PORT_FRONTEND
elif command -v gnome-open >/dev/null 2>&1; then
  gnome-open http://localhost:$PORT_FRONTEND
elif command -v open >/dev/null 2>&1; then
  open http://localhost:$PORT_FRONTEND
else
  echo "[INFO] Browser couldn't be opened automatically."
  echo "[INFO] Please open http://localhost:$PORT_FRONTEND in your browser."
fi

# Wait for the concurrently process
wait $CONCURRENTLY_PID

# Return success
exit 0
