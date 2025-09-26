#!/usr/bin/env python3
"""
Model Downloader for LocalLLM

Automatically downloads and sets up Qwen2.5-Coder-7B or StarCoder2-7B models.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import json
import logging
from typing import Dict, Optional
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Available models
AVAILABLE_MODELS = {
    'qwen2.5-coder-7b-instruct': {
        'hf_name': 'Qwen/Qwen2.5-Coder-7B-Instruct',
        'description': 'Qwen2.5 Coder 7B - Excellent for coding tasks',
        'size': '7B parameters (~14GB)',
        'memory_req': '16GB+ RAM recommended',
        'speciality': 'coding',
        'license': 'Apache 2.0',
        'recommended': True
    },
    'starcoder2-7b': {
        'hf_name': 'bigcode/starcoder2-7b',
        'description': 'StarCoder2 7B - Advanced code generation',
        'size': '7B parameters (~14GB)', 
        'memory_req': '16GB+ RAM recommended',
        'speciality': 'coding',
        'license': 'BigCode OpenRAIL-M',
        'recommended': True
    }
}

def check_dependencies():
    """Check if required dependencies are installed"""
    logger.info("🔍 Checking dependencies...")
    
    required_packages = [
        'torch',
        'transformers',
        'accelerate',
        'bitsandbytes'  # For quantization
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"❌ {package} - missing")
    
    if missing_packages:
        logger.info(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        
        for package in missing_packages:
            try:
                if package == 'bitsandbytes':
                    # Special handling for bitsandbytes
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", 
                        "bitsandbytes", "--quiet"
                    ])
                else:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", 
                        package, "--quiet"
                    ])
                logger.info(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️ Failed to install {package}: {e}")
    
    logger.info("✅ Dependencies check completed")

def get_model_cache_dir(model_name: str) -> Path:
    """Get the cache directory for a model"""
    cache_dir = Path.home() / '.cache' / 'localllm' / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def check_disk_space(required_gb: float) -> bool:
    """Check if there's enough disk space"""
    try:
        cache_dir = Path.home() / '.cache' / 'localllm'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        stat = shutil.disk_usage(cache_dir)
        free_gb = stat.free / (1024**3)
        
        logger.info(f"💾 Available disk space: {free_gb:.1f}GB")
        logger.info(f"💾 Required space: {required_gb:.1f}GB")
        
        if free_gb < required_gb:
            logger.error(f"❌ Insufficient disk space! Need {required_gb:.1f}GB, have {free_gb:.1f}GB")
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Could not check disk space: {e}")
        return True  # Assume it's okay if we can't check

def download_model(model_name: str, force_download: bool = False) -> bool:
    """Download a model using Hugging Face transformers"""
    
    if model_name not in AVAILABLE_MODELS:
        logger.error(f"❌ Unknown model: {model_name}")
        logger.info(f"Available models: {list(AVAILABLE_MODELS.keys())}")
        return False
    
    model_info = AVAILABLE_MODELS[model_name]
    hf_name = model_info['hf_name']
    
    logger.info(f"🤖 Downloading model: {model_name}")
    logger.info(f"   HuggingFace: {hf_name}")
    logger.info(f"   Description: {model_info['description']}")
    logger.info(f"   Size: {model_info['size']}")
    logger.info(f"   License: {model_info['license']}")
    
    # Check disk space (estimate 20GB needed)
    if not check_disk_space(20.0):
        return False
    
    # Get cache directory
    cache_dir = get_model_cache_dir(model_name)
    
    # Check if already downloaded
    if not force_download and (cache_dir / 'config.json').exists():
        logger.info(f"✅ Model already downloaded to: {cache_dir}")
        return True
    
    try:
        # Import here to avoid issues if not installed
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            hf_name,
            cache_dir=str(cache_dir),
            trust_remote_code=True
        )
        
        logger.info("📥 Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            cache_dir=str(cache_dir),
            trust_remote_code=True,
            torch_dtype='auto',
            low_cpu_mem_usage=True
        )
        
        # Save model info
        model_info_path = cache_dir / 'localllm_info.json'
        with open(model_info_path, 'w') as f:
            json.dump({
                'model_name': model_name,
                'hf_name': hf_name,
                'download_date': str(Path().cwd()),
                'model_info': model_info
            }, f, indent=2)
        
        logger.info(f"✅ Model downloaded successfully!")
        logger.info(f"   Location: {cache_dir}")
        logger.info(f"   Size: {sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()) / 1e9:.1f}GB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        
        # Clean up partial download
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                logger.info("🧹 Cleaned up partial download")
            except Exception:
                pass
        
        return False

def list_models():
    """List all available models"""
    print("\n🤖 Available Models for LocalLLM:")
    print("=" * 60)
    
    for model_name, info in AVAILABLE_MODELS.items():
        status = "⭐ RECOMMENDED" if info.get('recommended') else ""
        print(f"\n📦 {model_name} {status}")
        print(f"   Description: {info['description']}")
        print(f"   Size: {info['size']}")
        print(f"   Memory: {info['memory_req']}")
        print(f"   Specialty: {info['speciality']}")
        print(f"   License: {info['license']}")
        print(f"   HuggingFace: {info['hf_name']}")
        
        # Check if downloaded
        cache_dir = get_model_cache_dir(model_name)
        if (cache_dir / 'config.json').exists():
            print(f"   Status: ✅ Downloaded to {cache_dir}")
        else:
            print(f"   Status: ⬇️ Not downloaded")

def list_downloaded_models():
    """List downloaded models"""
    print("\n💾 Downloaded Models:")
    print("=" * 40)
    
    cache_base = Path.home() / '.cache' / 'localllm'
    
    if not cache_base.exists():
        print("No models downloaded yet.")
        return
    
    found_models = False
    
    for model_dir in cache_base.iterdir():
        if model_dir.is_dir() and (model_dir / 'config.json').exists():
            found_models = True
            
            # Get size
            try:
                size_gb = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / 1e9
            except:
                size_gb = 0
            
            print(f"✅ {model_dir.name}")
            print(f"   Location: {model_dir}")
            print(f"   Size: {size_gb:.1f}GB")
            
            # Load model info if available
            info_file = model_dir / 'localllm_info.json'
            if info_file.exists():
                try:
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    print(f"   Description: {info.get('model_info', {}).get('description', 'Unknown')}")
                except:
                    pass
            print()
    
    if not found_models:
        print("No models downloaded yet.")

def delete_model(model_name: str) -> bool:
    """Delete a downloaded model"""
    cache_dir = get_model_cache_dir(model_name)
    
    if not cache_dir.exists():
        logger.warning(f"⚠️ Model {model_name} not found")
        return False
    
    try:
        size_gb = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()) / 1e9
        
        logger.info(f"🗑️ Deleting model: {model_name}")
        logger.info(f"   Location: {cache_dir}")
        logger.info(f"   Size: {size_gb:.1f}GB")
        
        shutil.rmtree(cache_dir)
        
        logger.info(f"✅ Model {model_name} deleted successfully")
        logger.info(f"   Freed {size_gb:.1f}GB of disk space")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to delete model: {e}")
        return False

def test_model(model_name: str) -> bool:
    """Test if a model can be loaded"""
    if model_name not in AVAILABLE_MODELS:
        logger.error(f"❌ Unknown model: {model_name}")
        return False
    
    cache_dir = get_model_cache_dir(model_name)
    
    if not (cache_dir / 'config.json').exists():
        logger.error(f"❌ Model {model_name} not downloaded")
        return False
    
    try:
        logger.info(f"🧪 Testing model: {model_name}")
        
        # Add the localllm directory to path
        sys.path.append(str(Path(__file__).parent))
        
        from localllm import LocalLLM
        
        # Initialize model
        llm = LocalLLM(model_name=model_name, model_path=str(cache_dir))
        
        # Test generation
        test_prompt = "def fibonacci(n):"
        logger.info(f"   Test prompt: '{test_prompt}'")
        
        response = llm.generate(test_prompt, max_tokens=100)
        logger.info(f"   Response: '{response[:100]}...'")
        
        # Get model info
        info = llm.get_model_info()
        logger.info(f"   Parameters: {info['total_parameters']}")
        logger.info(f"   Device: {info['device']}")
        
        logger.info("✅ Model test successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="LocalLLM Model Downloader")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download a model')
    download_parser.add_argument('model', choices=list(AVAILABLE_MODELS.keys()), 
                               help='Model to download')
    download_parser.add_argument('--force', action='store_true', 
                               help='Force re-download even if model exists')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available models')
    
    # Downloaded command
    downloaded_parser = subparsers.add_parser('downloaded', help='List downloaded models')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a downloaded model')
    delete_parser.add_argument('model', help='Model to delete')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test a downloaded model')
    test_parser.add_argument('model', help='Model to test')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Check dependencies for download/test commands
    if args.command in ['download', 'test']:
        check_dependencies()
    
    if args.command == 'download':
        success = download_model(args.model, args.force)
        if success:
            print(f"\n🎉 Model {args.model} ready for use!")
            print(f"   Start chatting: python chat.py --model {args.model}")
        else:
            print(f"\n❌ Failed to download {args.model}")
            sys.exit(1)
    
    elif args.command == 'list':
        list_models()
    
    elif args.command == 'downloaded':
        list_downloaded_models()
    
    elif args.command == 'delete':
        success = delete_model(args.model)
        if not success:
            sys.exit(1)
    
    elif args.command == 'test':
        success = test_model(args.model)
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()
