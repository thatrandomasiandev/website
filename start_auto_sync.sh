#!/bin/bash

# Start Auto-Sync: Real-time GitHub synchronization
echo "🚀 Starting Auto-Sync..."

# Check if required packages are installed
if ! python3 -c "import watchdog" 2>/dev/null; then
    echo "📦 Installing required packages..."
    pip3 install watchdog
fi

# Start the auto-sync system
echo "🔄 Auto-sync is starting..."
python3 auto_sync.py
