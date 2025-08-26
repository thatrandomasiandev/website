#!/bin/bash

# Start HTTP Server for Personal Website
echo "🌐 Starting HTTP Server..."

# Start the HTTP server
echo "🚀 Starting HTTP server on http://localhost:8000"
cd core && python3 -m http.server 8000
