# ☄️ CometAI - Your Personal AI Assistant

A lightweight, downloadable Large Language Model that runs entirely on your local machine. No internet required, complete privacy, and easy to install.

## ✨ Features

- 🏠 **100% Local** - Runs entirely offline on your machine
- 🔒 **Complete Privacy** - Your conversations never leave your device
- ⚡ **Fast Setup** - One-click installer for all platforms
- 💻 **CPU Optimized** - Works on any modern computer (GPU optional)
- 🎯 **Lightweight** - Small download size, efficient memory usage
- 💬 **Chat Interface** - Simple, clean chat UI
- 🔧 **Customizable** - Adjust settings for your hardware

## 🚀 Quick Start

### Option 1: One-Click Installer
```bash
# Download and run the installer
curl -sSL https://raw.githubusercontent.com/your-repo/LocalLLM/main/install.sh | bash
```

### Option 2: Manual Installation
```bash
# Clone the repository
git clone https://github.com/your-repo/CometAI.git
cd CometAI

# Run the setup
python setup.py install

# Start chatting
python chat.py
```

## 💾 System Requirements

### Minimum Requirements
- **RAM**: 4GB available
- **Storage**: 2GB free space
- **CPU**: Any modern processor (2015+)
- **OS**: Windows 10+, macOS 10.14+, Linux

### Recommended Requirements
- **RAM**: 8GB+ available
- **Storage**: 5GB+ free space
- **CPU**: Multi-core processor
- **GPU**: Optional (NVIDIA/AMD for faster inference)

## 📱 Usage

### Chat Interface
```bash
# Start the chat interface
python chat.py

# Or use the GUI version
python gui_chat.py
```

### Command Line
```bash
# Single question
python ask.py "What is artificial intelligence?"

# Interactive mode
python ask.py --interactive
```

### API Mode
```bash
# Start local API server
python api_server.py

# Use the API
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
model:
  size: "small"  # small, medium, large
  max_tokens: 2048
  temperature: 0.7

performance:
  use_gpu: auto  # true, false, auto
  threads: auto  # number of CPU threads
  memory_limit: "4GB"

interface:
  theme: "dark"  # dark, light
  save_history: true
```

## 🔧 Advanced Usage

### Custom Models
```bash
# Load a different model
python chat.py --model path/to/your/model.bin

# Fine-tune on your data
python train.py --data your_data.txt --output custom_model
```

### Integration
```python
from localllm import LocalLLM

# Initialize CometAI
comet = LocalLLM()

# Generate text
response = comet.generate("Tell me a joke")
print(response)

# Chat conversation
conversation = comet.chat()
conversation.send("Hello!")
```

## 📦 What's Included

- **Pre-trained Model** - Ready-to-use language model
- **Chat Interface** - Terminal and GUI chat applications
- **API Server** - REST API for integration
- **Documentation** - Complete setup and usage guides
- **Examples** - Sample code and use cases

## 🛠️ Development

### Building from Source
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Build the model
python build_model.py

# Run tests
python -m pytest tests/

# Create distribution
python setup.py sdist bdist_wheel
```

## 📄 License

MIT License - Use freely for personal and commercial projects.

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📞 Support

- 📖 **Documentation**: [docs/](docs/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-repo/LocalLLM/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-repo/LocalLLM/discussions)

---

**Made with ❤️ for privacy-conscious AI users**