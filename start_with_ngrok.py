#!/usr/bin/env python3
"""
CometAI Server with ngrok Tunnel

Alternative tunnel solution using ngrok for public access.
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def check_ngrok():
    """Check if ngrok is installed"""
    try:
        result = subprocess.run(['ngrok', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ ngrok found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ ngrok not found")
    return False

def install_ngrok():
    """Install ngrok"""
    print("📦 Installing ngrok...")
    print("Please download ngrok from: https://ngrok.com/download")
    print("Or install via package manager:")
    print("  - Windows: winget install ngrok.ngrok")
    print("  - Mac: brew install ngrok")
    print("  - Linux: snap install ngrok")
    return False

def start_ngrok_tunnel(port=8080):
    """Start ngrok tunnel"""
    print(f"🌐 Starting ngrok tunnel on port {port}...")
    
    try:
        # Start ngrok in background
        process = subprocess.Popen(
            ['ngrok', 'http', str(port), '--log=stdout'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for tunnel to start
        time.sleep(3)
        
        # Get tunnel URL from ngrok API
        try:
            response = requests.get('http://localhost:4040/api/tunnels')
            if response.status_code == 200:
                tunnels = response.json()['tunnels']
                if tunnels:
                    tunnel_url = tunnels[0]['public_url']
                    print(f"✅ ngrok tunnel started: {tunnel_url}")
                    return tunnel_url, process
        except:
            print("⚠️  Could not get tunnel URL from ngrok API")
            print("Check ngrok web interface at: http://localhost:4040")
            return None, process
        
    except Exception as e:
        print(f"❌ Failed to start ngrok: {e}")
        return None, None

def start_server():
    """Start the AI server"""
    print("🚀 Starting CometAI server...")
    
    try:
        # Import and start the server
        from unified_web_server import UnifiedCometAIServer
        
        server = UnifiedCometAIServer(
            host='127.0.0.1',  # Bind to localhost for ngrok
            port=8080,
            model_name='qwen2.5-coder-7b-instruct'
        )
        
        print("🤖 Server ready - starting in background...")
        
        # Start server in background thread
        import threading
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        time.sleep(5)
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

def main():
    """Main function"""
    print("☄️ CometAI with ngrok Tunnel")
    print("=" * 35)
    
    # Check if ngrok is available
    if not check_ngrok():
        if not install_ngrok():
            sys.exit(1)
    
    # Start the server
    if not start_server():
        sys.exit(1)
    
    # Start ngrok tunnel
    tunnel_url, ngrok_process = start_ngrok_tunnel()
    
    if not tunnel_url:
        print("❌ Failed to start tunnel")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 CometAI Server is Live!")
    print(f"🌐 Public URL: {tunnel_url}")
    print(f"🤖 AI API: {tunnel_url}/api")
    print(f"📊 Health Check: {tunnel_url}/api/health")
    print("\n📋 ngrok Dashboard: http://localhost:4040")
    print("Press Ctrl+C to stop both server and tunnel")
    print("=" * 50)
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        if ngrok_process:
            ngrok_process.terminate()
        print("✅ Server and tunnel stopped")

if __name__ == "__main__":
    main()
