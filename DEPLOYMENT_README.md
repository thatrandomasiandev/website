# CometAI Server Deployment Guide

## 🚀 Quick Start

Your AI server is ready for deployment! Here's how to get it running with a dev tunnel.

### 1. Start the Production Server

Choose one of these options:

**Option A: Unified Server (Website + AI)**
```bash
python start_unified_server_production.py
```

**Option B: AI-Only Server**
```bash
python start_ai_server_production.py
```

### 2. Configure Dev Tunnel

Once you have your dev tunnel URL, configure it:

```bash
python tunnel_helper.py set-url <your_tunnel_url>
```

### 3. Check Status

```bash
python tunnel_helper.py status
```

## 📋 What's Ready

✅ **AI Model**: Qwen2.5-Coder-7B (7.6B parameters) loaded and optimized
✅ **Server**: Production-ready Flask server configured for 0.0.0.0 binding
✅ **Dependencies**: All required packages installed and tested
✅ **API Endpoints**: Full REST API for chat functionality
✅ **Tunnel Helper**: Automated configuration management

## 🔗 API Endpoints

Once running, your server will provide:

- `GET /` - Main website (unified server only)
- `GET /api` - API information
- `GET /api/health` - Health check
- `GET /api/model/status` - Model loading status
- `POST /api/chat` - Chat with AI
- `POST /api/chat/new` - Start new conversation
- `GET /api/chat/history/<session_id>` - Get conversation history
- `DELETE /api/chat/clear/<session_id>` - Clear conversation

## 🛠️ Configuration

The server uses environment variables:
- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 8080)
- `MODEL_NAME` - AI model to use (default: qwen2.5-coder-7b-instruct)

## 📝 Chat API Usage

```bash
# Start a chat
curl -X POST http://your-tunnel-url/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'

# Health check
curl http://your-tunnel-url/api/health
```

## 🔧 Troubleshooting

1. **Model loading issues**: Check available RAM (requires 14GB+)
2. **Port conflicts**: Change PORT environment variable
3. **Dependencies**: Run `pip install -r requirements.txt`

## 📁 Files Created

- `start_ai_server_production.py` - AI-only production server
- `start_unified_server_production.py` - Website + AI production server
- `tunnel_helper.py` - Dev tunnel configuration helper
- `tunnel_config.json` - Tunnel configuration (auto-created)
- `.env` - Environment variables (auto-created)
