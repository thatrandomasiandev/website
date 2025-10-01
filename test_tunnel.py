#!/usr/bin/env python3
"""
Test Dev Tunnel Connectivity

Test script to diagnose tunnel connectivity issues.
"""

import requests
import time
import subprocess
import sys

def test_local_server():
    """Test if local server is responding"""
    print("🔍 Testing local server...")
    
    try:
        response = requests.get('http://localhost:8080/api/health', timeout=5)
        if response.status_code == 200:
            print("✅ Local server is responding")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Local server returned status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Local server not responding: {e}")
        return False

def test_tunnel_url(tunnel_url):
    """Test tunnel URL"""
    print(f"🔍 Testing tunnel URL: {tunnel_url}")
    
    # Test different endpoints
    endpoints = [
        f"{tunnel_url}",
        f"{tunnel_url}/",
        f"{tunnel_url}/api",
        f"{tunnel_url}/api/health",
        f"{tunnel_url}/api/",
    ]
    
    for endpoint in endpoints:
        try:
            print(f"   Testing: {endpoint}")
            response = requests.get(endpoint, timeout=10)
            print(f"   ✅ Status: {response.status_code}")
            if response.status_code == 200:
                print(f"   📄 Content preview: {response.text[:200]}...")
                return endpoint
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return None

def check_tunnel_process():
    """Check if tunnel process is running"""
    print("🔍 Checking tunnel process...")
    
    try:
        # Check for dev tunnel process
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq devtunnel.exe'], 
                              capture_output=True, text=True)
        if 'devtunnel.exe' in result.stdout:
            print("✅ Dev tunnel process is running")
            return True
        else:
            print("❌ Dev tunnel process not found")
            return False
    except Exception as e:
        print(f"❌ Error checking tunnel process: {e}")
        return False

def main():
    """Main function"""
    print("🧪 CometAI Tunnel Diagnostic Tool")
    print("=" * 40)
    
    # Test local server
    if not test_local_server():
        print("\n❌ Local server is not responding. Please start the server first.")
        return
    
    # Check tunnel process
    tunnel_running = check_tunnel_process()
    
    # Test tunnel URL
    tunnel_url = "https://cz6cnk10-8080.usw3.devtunnels.ms"
    working_endpoint = test_tunnel_url(tunnel_url)
    
    print("\n" + "=" * 40)
    print("📊 DIAGNOSTIC RESULTS:")
    print(f"   Local Server: {'✅ Working' if test_local_server() else '❌ Not Working'}")
    print(f"   Tunnel Process: {'✅ Running' if tunnel_running else '❌ Not Running'}")
    print(f"   Tunnel Access: {'✅ Working' if working_endpoint else '❌ Not Working'}")
    
    if working_endpoint:
        print(f"\n🎉 Working endpoint: {working_endpoint}")
    else:
        print("\n🔧 TROUBLESHOOTING SUGGESTIONS:")
        print("   1. Check if dev tunnel is set to 'Public' access")
        print("   2. Verify tunnel is properly configured")
        print("   3. Try restarting the dev tunnel")
        print("   4. Consider using ngrok as alternative")
        print("   5. Check firewall/antivirus settings")

if __name__ == "__main__":
    main()
