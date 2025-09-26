#!/usr/bin/env python3
"""
CometAI Startup Script

Easy startup script for the CometAI web API server.
Handles dependency checking and provides helpful error messages.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        ('flask', 'Flask'),
        ('flask_cors', 'Flask-CORS'),
        ('transformers', 'Transformers'),
        ('torch', 'PyTorch'),
        ('yaml', 'PyYAML')
    ]
    
    missing_packages = []
    
    for package, display_name in required_packages:
        try:
            __import__(package)
            print(f"✅ {display_name} is installed")
        except ImportError:
            missing_packages.append((package, display_name))
            print(f"❌ {display_name} is missing")
    
    return missing_packages

def install_dependencies(missing_packages):
    """Install missing dependencies"""
    if not missing_packages:
        return True
    
    print(f"\n📦 Installing {len(missing_packages)} missing packages...")
    
    # Create pip install command
    packages_to_install = []
    for package, display_name in missing_packages:
        if package == 'flask_cors':
            packages_to_install.append('flask-cors')
        elif package == 'yaml':
            packages_to_install.append('pyyaml')
        else:
            packages_to_install.append(package)
    
    try:
        cmd = [sys.executable, '-m', 'pip', 'install'] + packages_to_install
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_model_availability():
    """Check if models are available"""
    print("\n🤖 Checking model availability...")
    
    # Check if models directory exists
    models_dir = Path.home() / '.cache' / 'huggingface' / 'transformers'
    
    if models_dir.exists():
        print(f"✅ HuggingFace cache directory found: {models_dir}")
    else:
        print("⚠️  No cached models found. Models will be downloaded on first use.")
    
    return True

def start_server():
    """Start the CometAI web API server"""
    print("\n🚀 Starting CometAI Web API Server...")
    print("=" * 50)
    
    try:
        # Import and start the server
        from web_api_server import CometAIWebServer
        
        server = CometAIWebServer(
            host='localhost',
            port=8080,
            model_name='qwen2.5-coder-7b-instruct'
        )
        
        print("🌐 Server starting at: http://localhost:8080")
        print("💬 Chat interface at: http://localhost:3000/chat-interface.html")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 50)
        
        server.run(debug=False)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("☄️ CometAI Startup Script")
    print("=" * 30)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"\n⚠️  {len(missing_packages)} required packages are missing.")
        response = input("Would you like to install them automatically? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            if not install_dependencies(missing_packages):
                print("\n❌ Failed to install dependencies. Please install them manually:")
                for package, display_name in missing_packages:
                    if package == 'flask_cors':
                        print(f"  pip install flask-cors")
                    elif package == 'yaml':
                        print(f"  pip install pyyaml")
                    else:
                        print(f"  pip install {package}")
                sys.exit(1)
        else:
            print("\n❌ Cannot start without required dependencies.")
            sys.exit(1)
    
    # Check model availability
    check_model_availability()
    
    # Start the server
    if not start_server():
        sys.exit(1)

if __name__ == "__main__":
    main()
