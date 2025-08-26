#!/usr/bin/env python3
"""
HTTPS Server for Personal Website
Serves the website with SSL encryption for local development
"""

import http.server
import socketserver
import ssl
import os
from pathlib import Path

# Configuration
PORT = 8443
CERT_FILE = 'ssl/cert.pem'
KEY_FILE = 'ssl/key.pem'

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add security headers
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('X-XSS-Protection', '1; mode=block')
        self.send_header('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
        super().end_headers()

def main():
    # Check if SSL files exist
    if not os.path.exists(CERT_FILE) or not os.path.exists(KEY_FILE):
        print(f"Error: SSL certificate files not found!")
        print(f"Expected: {CERT_FILE} and {KEY_FILE}")
        print("Run the setup script first to generate SSL certificates.")
        return
    
    # Create HTTPS server
    httpd = socketserver.TCPServer(("", PORT), MyHTTPRequestHandler)
    
    # Wrap socket with SSL
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(CERT_FILE, KEY_FILE)
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    
    print(f"🚀 HTTPS Server running on https://localhost:{PORT}")
    print(f"📁 Serving files from: {os.getcwd()}")
    print(f"🔒 SSL Certificate: {CERT_FILE}")
    print(f"🔑 Private Key: {KEY_FILE}")
    print("\n⚠️  Note: This is a self-signed certificate.")
    print("   Your browser will show a security warning - this is normal for local development.")
    print("   Click 'Advanced' → 'Proceed to localhost (unsafe)' to continue.")
    print("\n🔄 Press Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        httpd.shutdown()

if __name__ == "__main__":
    main()
