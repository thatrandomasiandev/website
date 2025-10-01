#!/usr/bin/env python3
"""
Dev Tunnel Helper Script

Helper script to configure and manage dev tunnel integration for CometAI.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

class TunnelHelper:
    """Helper class for managing dev tunnel integration"""
    
    def __init__(self):
        self.config_file = Path("tunnel_config.json")
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load tunnel configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load tunnel config: {e}")
        
        return {
            "tunnel_url": None,
            "server_host": "0.0.0.0",
            "server_port": 8080,
            "server_type": "unified",  # "unified" or "ai_only"
            "model_name": "qwen2.5-coder-7b-instruct"
        }
    
    def save_config(self):
        """Save tunnel configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"✅ Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"❌ Failed to save config: {e}")
    
    def set_tunnel_url(self, tunnel_url: str):
        """Set the dev tunnel URL"""
        self.config["tunnel_url"] = tunnel_url
        self.save_config()
        print(f"✅ Tunnel URL set to: {tunnel_url}")
    
    def get_tunnel_url(self) -> Optional[str]:
        """Get the current tunnel URL"""
        return self.config.get("tunnel_url")
    
    def configure_server(self, server_type: str = "unified", port: int = 8080):
        """Configure server settings"""
        self.config["server_type"] = server_type
        self.config["server_port"] = port
        self.save_config()
        print(f"✅ Server configured: {server_type} on port {port}")
    
    def get_startup_command(self) -> str:
        """Get the appropriate startup command based on configuration"""
        server_type = self.config.get("server_type", "unified")
        
        if server_type == "unified":
            return "python start_unified_server_production.py"
        else:
            return "python start_ai_server_production.py"
    
    def print_status(self):
        """Print current tunnel and server status"""
        print("\n🔗 Dev Tunnel Status")
        print("=" * 25)
        print(f"Tunnel URL: {self.config.get('tunnel_url', 'Not set')}")
        print(f"Server Type: {self.config.get('server_type', 'unified')}")
        print(f"Server Port: {self.config.get('server_port', 8080)}")
        print(f"Model: {self.config.get('model_name', 'qwen2.5-coder-7b-instruct')}")
        print(f"Startup Command: {self.get_startup_command()}")
        
        if self.config.get("tunnel_url"):
            print(f"\n🌐 Access URLs:")
            print(f"  Website: {self.config['tunnel_url']}")
            print(f"  AI API: {self.config['tunnel_url']}/api")
            print(f"  Health Check: {self.config['tunnel_url']}/api/health")
    
    def create_env_file(self):
        """Create .env file for production deployment"""
        env_content = f"""# CometAI Production Environment Variables
HOST=0.0.0.0
PORT={self.config.get('server_port', 8080)}
MODEL_NAME={self.config.get('model_name', 'qwen2.5-coder-7b-instruct')}

# Dev Tunnel Configuration
TUNNEL_URL={self.config.get('tunnel_url', '')}

# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=False
"""
        
        env_file = Path(".env")
        try:
            with open(env_file, 'w') as f:
                f.write(env_content)
            print(f"✅ Environment file created: {env_file}")
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")

def main():
    """Main function for command-line usage"""
    helper = TunnelHelper()
    
    if len(sys.argv) < 2:
        print("🔗 CometAI Dev Tunnel Helper")
        print("=" * 30)
        print("Usage:")
        print("  python tunnel_helper.py set-url <tunnel_url>")
        print("  python tunnel_helper.py configure <server_type> <port>")
        print("  python tunnel_helper.py status")
        print("  python tunnel_helper.py create-env")
        print("\nServer types: 'unified' (website + AI) or 'ai_only' (AI only)")
        print("\nCurrent status:")
        helper.print_status()
        return
    
    command = sys.argv[1].lower()
    
    if command == "set-url" and len(sys.argv) > 2:
        tunnel_url = sys.argv[2]
        helper.set_tunnel_url(tunnel_url)
        helper.create_env_file()
        
    elif command == "configure" and len(sys.argv) > 3:
        server_type = sys.argv[2]
        port = int(sys.argv[3])
        helper.configure_server(server_type, port)
        
    elif command == "status":
        helper.print_status()
        
    elif command == "create-env":
        helper.create_env_file()
        
    else:
        print("❌ Invalid command or missing arguments")
        print("Use 'python tunnel_helper.py' for help")

if __name__ == "__main__":
    main()
