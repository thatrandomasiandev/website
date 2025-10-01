#!/usr/bin/env python3
"""
CometAI Unified Production Server Startup Script

Production-ready startup script for the unified CometAI server.
Serves both website and AI functionality, configured for cloud deployment.
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

def start_unified_production_server():
    """Start the unified CometAI server in production mode"""
    print("\n🚀 Starting CometAI Unified Production Server...")
    print("=" * 55)
    
    # Get configuration from environment variables or use defaults
    host = os.environ.get('HOST', '0.0.0.0')  # Bind to all interfaces for tunnel access
    port = int(os.environ.get('PORT', 8080))
    model_name = os.environ.get('MODEL_NAME', 'qwen2.5-coder-7b-instruct')
    
    try:
        # Import and start the unified server
        from unified_web_server import UnifiedCometAIServer
        
        server = UnifiedCometAIServer(
            host=host,
            port=port,
            model_name=model_name
        )
        
        print(f"🌐 Website: http://{host}:{port}")
        print(f"🤖 AI API: http://{host}:{port}/api")
        print(f"🤖 Model: {model_name}")
        print(f"🔗 Dev tunnel ready - waiting for tunnel URL...")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 55)
        
        server.run(debug=False)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("☄️ CometAI Unified Production Server")
    print("=" * 40)
    
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
    
    # Start the unified production server
    if not start_unified_production_server():
        sys.exit(1)

if __name__ == "__main__":
    main()
