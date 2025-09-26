# CometAI Integration Guide

Welcome to CometAI - your personal AI coding assistant integrated into Joshua Terranova's website!

## 🚀 Quick Start

### 1. Install Dependencies

First, make sure you have Python 3.8+ installed, then install the required packages:

```bash
pip install flask flask-cors transformers torch pyyaml
```

Or use the automatic installer:

```bash
python start_comet_api.py
```

### 2. Start the API Server

Run the CometAI web API server:

```bash
python web_api_server.py
```

Or use the startup script:

```bash
python start_comet_api.py
```

### 3. Access the Chat Interface

Open your web browser and navigate to:
- **Chat Interface**: `http://localhost:3000/chat-interface.html`
- **AI Lab Page**: `http://localhost:3000/ai-lab.html`
- **Main Website**: `http://localhost:3000/index.html`

## 📁 File Structure

```
Personal Website/
├── web_api_server.py          # Flask API server for CometAI
├── chat-interface.html        # Modern web chat interface
├── start_comet_api.py         # Easy startup script
├── localllm/                  # Local LLM implementation
│   ├── __init__.py
│   ├── model.py              # Main LLM model class
│   ├── chat.py               # Chat interface
│   └── api.py                # API wrapper
├── config.yaml               # Model configuration
├── chat.py                   # Command-line chat interface
├── gui_chat.py               # Desktop GUI chat interface
└── requirements.txt          # Python dependencies
```

## 🤖 Available Models

CometAI supports these pre-trained models:

1. **Qwen2.5-Coder-7B-Instruct** (Default)
   - Excellent for coding tasks
   - 7B parameters
   - Requires ~14GB RAM

2. **StarCoder2-7B**
   - Advanced code generation
   - 7B parameters
   - Requires ~14GB RAM

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
model:
  name: "qwen2.5-coder-7b-instruct"  # or "starcoder2-7b"
  max_tokens: 512
  temperature: 0.7
  
performance:
  use_gpu: "auto"
  load_in_8bit: true
  torch_dtype: "bfloat16"
```

## 🌐 API Endpoints

The web API server provides these endpoints:

- `GET /` - API information
- `GET /health` - Health check
- `GET /model/info` - Model information
- `GET /model/status` - Model loading status
- `POST /chat` - Chat with AI
- `POST /chat/new` - Start new conversation
- `GET /chat/history/<session_id>` - Get conversation history
- `DELETE /chat/clear/<session_id>` - Clear conversation

### Example API Usage

```javascript
// Start a new chat
const response = await fetch('http://localhost:8080/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        message: "Hello, can you help me with Python?",
        session_id: "my-session-123",
        max_tokens: 512,
        temperature: 0.7
    })
});

const data = await response.json();
console.log(data.response);
```

## 💻 Interface Options

### 1. Web Chat Interface
- Modern, responsive design
- Real-time messaging
- Code syntax highlighting
- Session management
- Access: `chat-interface.html`

### 2. Command Line Interface
```bash
python chat.py --model qwen2.5-coder-7b-instruct
```

### 3. Desktop GUI
```bash
python gui_chat.py
```

## 🔍 Troubleshooting

### Model Loading Issues

1. **Out of Memory**: Enable quantization in `config.yaml`:
   ```yaml
   performance:
     load_in_8bit: true
   ```

2. **Model Download**: Models are downloaded automatically on first use. Ensure you have internet connection and sufficient disk space (~15GB per model).

3. **GPU Issues**: If GPU causes problems, force CPU usage:
   ```yaml
   performance:
     use_gpu: false
   ```

### Web Interface Issues

1. **Server Not Available**: Make sure the API server is running on port 8080
2. **CORS Errors**: The server includes CORS headers, but ensure you're accessing from the correct domain
3. **Connection Refused**: Check firewall settings and ensure port 8080 is available

### Performance Optimization

For Apple Silicon (M1/M2/M3/M4):
```yaml
performance:
  torch_dtype: "bfloat16"
  use_mps: true
  enable_torch_compile: true
```

For CUDA GPUs:
```yaml
performance:
  torch_dtype: "float16"
  use_gpu: true
  device_map: "auto"
```

## 🛡️ Privacy & Security

- **Local Processing**: All AI processing happens locally on your machine
- **No Data Sharing**: Conversations are not sent to external servers
- **Session Storage**: Chat history is stored locally and can be cleared anytime
- **Open Source**: Full source code is available for inspection

## 📊 System Requirements

### Minimum Requirements
- Python 3.8+
- 16GB RAM
- 20GB free disk space
- Internet connection (for initial model download)

### Recommended Requirements
- Python 3.10+
- 32GB RAM
- GPU with 8GB+ VRAM (optional but recommended)
- SSD storage

## 🔄 Updates & Maintenance

### Updating Models
Models are cached in `~/.cache/huggingface/transformers/`. To update:
1. Delete the model cache directory
2. Restart the server to download the latest version

### Clearing Chat History
- Web Interface: Use the "Clear Chat" button
- API: `DELETE /chat/clear/<session_id>`
- Files: Delete `chat_history.json` and `gui_chat_history.json`

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the console output for error messages
3. Ensure all dependencies are properly installed
4. Check system requirements

## 📝 License

This CometAI integration is part of Joshua Terranova's personal website project.

---

**Enjoy chatting with CometAI! 🚀**
