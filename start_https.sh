#!/bin/bash

# Start HTTPS Server for Personal Website
echo "🔒 Starting HTTPS Server..."

# Check if SSL certificates exist
if [ ! -f "ssl/cert.pem" ] || [ ! -f "ssl/key.pem" ]; then
    echo "❌ SSL certificates not found!"
    echo "Generating SSL certificates..."
    
    # Create ssl directory if it doesn't exist
    mkdir -p ssl
    
    # Generate self-signed certificate
    openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes -subj "/C=US/ST=CA/L=Local/O=Development/CN=localhost"
    
    if [ $? -eq 0 ]; then
        echo "✅ SSL certificates generated successfully!"
    else
        echo "❌ Failed to generate SSL certificates!"
        exit 1
    fi
fi

# Start the HTTPS server
echo "🚀 Starting HTTPS server on https://localhost:8443"
python3 https_server.py
