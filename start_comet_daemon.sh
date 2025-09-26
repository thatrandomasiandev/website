#!/bin/bash

# CometAI Server Daemon Startup Script
# This script ensures the server runs continuously with auto-restart

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SCRIPT="$SCRIPT_DIR/web_api_server.py"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$SCRIPT_DIR/comet_server.pid"
LOG_FILE="$LOG_DIR/comet_server.log"
ERROR_LOG="$LOG_DIR/comet_server_error.log"
MAX_RESTARTS=10
RESTART_DELAY=5

# Create logs directory
mkdir -p "$LOG_DIR"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check if server is running
is_server_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Function to stop the server
stop_server() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        log_message "Stopping CometAI server (PID: $pid)"
        kill "$pid" 2>/dev/null
        sleep 2
        if ps -p "$pid" > /dev/null 2>&1; then
            log_message "Force killing server"
            kill -9 "$pid" 2>/dev/null
        fi
        rm -f "$PID_FILE"
    fi
}

# Function to start the server
start_server() {
    log_message "Starting CometAI server..."
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Start server in background and capture PID
    nohup python3 "$SERVER_SCRIPT" >> "$LOG_FILE" 2>> "$ERROR_LOG" &
    local server_pid=$!
    
    # Save PID
    echo "$server_pid" > "$PID_FILE"
    log_message "CometAI server started with PID: $server_pid"
    
    # Wait a moment to check if it started successfully
    sleep 3
    if ! ps -p "$server_pid" > /dev/null 2>&1; then
        log_message "ERROR: Server failed to start"
        rm -f "$PID_FILE"
        return 1
    fi
    
    return 0
}

# Function to monitor and restart server
monitor_server() {
    local restart_count=0
    
    log_message "Starting CometAI server monitoring daemon"
    
    while true; do
        if ! is_server_running; then
            if [ $restart_count -ge $MAX_RESTARTS ]; then
                log_message "ERROR: Maximum restart attempts ($MAX_RESTARTS) reached. Stopping daemon."
                break
            fi
            
            log_message "Server not running. Attempting restart ($((restart_count + 1))/$MAX_RESTARTS)"
            
            if start_server; then
                restart_count=0  # Reset counter on successful start
                log_message "Server restarted successfully"
            else
                restart_count=$((restart_count + 1))
                log_message "Failed to restart server. Waiting $RESTART_DELAY seconds..."
                sleep $RESTART_DELAY
            fi
        fi
        
        # Check every 30 seconds
        sleep 30
    done
}

# Handle script arguments
case "$1" in
    start)
        if is_server_running; then
            log_message "CometAI server is already running"
            exit 0
        fi
        start_server
        ;;
    stop)
        stop_server
        log_message "CometAI server stopped"
        ;;
    restart)
        stop_server
        sleep 2
        start_server
        ;;
    status)
        if is_server_running; then
            local pid=$(cat "$PID_FILE")
            log_message "CometAI server is running (PID: $pid)"
            # Test server health
            if curl -s http://localhost:8080/health > /dev/null; then
                log_message "Server health check: OK"
            else
                log_message "Server health check: FAILED"
            fi
        else
            log_message "CometAI server is not running"
        fi
        ;;
    monitor)
        # Run monitoring daemon
        monitor_server
        ;;
    logs)
        echo "=== Server Log ==="
        tail -n 50 "$LOG_FILE" 2>/dev/null || echo "No server log found"
        echo ""
        echo "=== Error Log ==="
        tail -n 20 "$ERROR_LOG" 2>/dev/null || echo "No error log found"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|monitor|logs}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the CometAI server"
        echo "  stop    - Stop the CometAI server"
        echo "  restart - Restart the CometAI server"
        echo "  status  - Check server status and health"
        echo "  monitor - Run monitoring daemon (auto-restart)"
        echo "  logs    - Show recent logs"
        exit 1
        ;;
esac
