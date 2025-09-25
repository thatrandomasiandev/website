#!/usr/bin/env python3
"""
CometAI Web API Server

Flask-based web API server for serving the LocalLLM model over HTTP.
Provides REST endpoints for chat functionality and model information.
"""

import os
import sys
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

# Flask imports
try:
    from flask import Flask, request, jsonify, render_template_string, send_from_directory
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Add localllm to path
sys.path.append(str(Path(__file__).parent))

try:
    from localllm import LocalLLM
    LOCALLLM_AVAILABLE = True
except ImportError:
    LOCALLLM_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class CometAIWebServer:
    """Web API server for CometAI"""
    
    def __init__(self, host='localhost', port=8080, model_name=None, config_path=None):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required. Install with: pip install flask flask-cors")
        
        if not LOCALLLM_AVAILABLE:
            raise ImportError("LocalLLM is required. Make sure it's properly installed.")
        
        self.host = host
        self.port = port
        self.model_name = model_name or 'qwen2.5-coder-7b-instruct'
        self.config_path = config_path
        
        # Initialize Flask app
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for web integration
        
        # Model and conversation storage
        self.llm = None
        self.conversations = {}  # Store conversations by session ID
        self.model_loading = False
        self.model_loaded = False
        self.model_error = None
        
        # Setup routes
        self._setup_routes()
        
        # Start model loading in background
        self._load_model_async()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main API info page"""
            return jsonify({
                'name': 'CometAI Web API',
                'version': '1.0.0',
                'description': 'Local AI Assistant API',
                'author': 'Joshua Terranova',
                'endpoints': {
                    '/': 'API information',
                    '/health': 'Health check',
                    '/model/info': 'Model information',
                    '/model/status': 'Model loading status',
                    '/chat': 'Chat with AI (POST)',
                    '/chat/new': 'Start new conversation (POST)',
                    '/chat/history/<session_id>': 'Get conversation history',
                    '/chat/clear/<session_id>': 'Clear conversation (DELETE)'
                },
                'status': 'running',
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/health')
        def health():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'model_loaded': self.model_loaded,
                'model_loading': self.model_loading,
                'model_error': self.model_error,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/model/info')
        def model_info():
            """Get model information"""
            if not self.model_loaded:
                return jsonify({
                    'error': 'Model not loaded',
                    'model_loading': self.model_loading,
                    'model_error': self.model_error
                }), 503
            
            try:
                info = self.llm.get_model_info()
                return jsonify({
                    'success': True,
                    'model_info': info,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error getting model info: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/model/status')
        def model_status():
            """Get model loading status"""
            return jsonify({
                'model_loaded': self.model_loaded,
                'model_loading': self.model_loading,
                'model_error': self.model_error,
                'model_name': self.model_name,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/chat', methods=['POST'])
        def chat():
            """Chat with the AI"""
            if not self.model_loaded:
                return jsonify({
                    'error': 'Model not loaded',
                    'model_loading': self.model_loading,
                    'model_error': self.model_error
                }), 503
            
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No JSON data provided'}), 400
                
                message = data.get('message', '').strip()
                session_id = data.get('session_id', str(uuid.uuid4()))
                max_tokens = data.get('max_tokens', 512)
                temperature = data.get('temperature', 0.7)
                
                if not message:
                    return jsonify({'error': 'No message provided'}), 400
                
                # Get or create conversation history
                if session_id not in self.conversations:
                    self.conversations[session_id] = []
                
                conversation_history = self.conversations[session_id]
                
                # Generate response
                logger.info(f"Generating response for session {session_id[:8]}...")
                response = self.llm.chat(
                    message,
                    conversation_history=conversation_history,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Update conversation history
                self.conversations[session_id].append({
                    'role': 'user',
                    'content': message,
                    'timestamp': datetime.now().isoformat()
                })
                self.conversations[session_id].append({
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Keep conversation history manageable (last 20 messages)
                if len(self.conversations[session_id]) > 20:
                    self.conversations[session_id] = self.conversations[session_id][-20:]
                
                return jsonify({
                    'success': True,
                    'response': response,
                    'session_id': session_id,
                    'message_count': len(self.conversations[session_id]),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Chat error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/chat/new', methods=['POST'])
        def new_chat():
            """Start a new conversation"""
            session_id = str(uuid.uuid4())
            self.conversations[session_id] = []
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': 'New conversation started',
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/chat/history/<session_id>')
        def chat_history(session_id):
            """Get conversation history"""
            if session_id not in self.conversations:
                return jsonify({'error': 'Session not found'}), 404
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'history': self.conversations[session_id],
                'message_count': len(self.conversations[session_id]),
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/chat/clear/<session_id>', methods=['DELETE'])
        def clear_chat(session_id):
            """Clear conversation history"""
            if session_id in self.conversations:
                del self.conversations[session_id]
                return jsonify({
                    'success': True,
                    'message': 'Conversation cleared',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({'error': 'Session not found'}), 404
        
        @self.app.route('/chat/sessions')
        def list_sessions():
            """List active chat sessions"""
            sessions = []
            for session_id, history in self.conversations.items():
                sessions.append({
                    'session_id': session_id,
                    'message_count': len(history),
                    'last_activity': history[-1]['timestamp'] if history else None
                })
            
            return jsonify({
                'success': True,
                'sessions': sessions,
                'total_sessions': len(sessions),
                'timestamp': datetime.now().isoformat()
            })
        
        # Error handlers
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'error': 'Endpoint not found',
                'message': 'The requested endpoint does not exist',
                'available_endpoints': [
                    '/', '/health', '/model/info', '/model/status',
                    '/chat', '/chat/new', '/chat/history/<session_id>',
                    '/chat/clear/<session_id>', '/chat/sessions'
                ]
            }), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({
                'error': 'Internal server error',
                'message': 'An unexpected error occurred'
            }), 500
    
    def _load_model_async(self):
        """Load the model in a background thread"""
        def load_model():
            try:
                self.model_loading = True
                self.model_error = None
                logger.info(f"Loading CometAI model: {self.model_name}")
                
                # Initialize the model
                self.llm = LocalLLM(
                    model_name=self.model_name,
                    config_path=self.config_path
                )
                
                # Apply optimizations
                self.llm.optimize_for_inference()
                
                self.model_loaded = True
                self.model_loading = False
                
                logger.info("✅ CometAI model loaded successfully!")
                logger.info(f"Model info: {self.llm.get_model_info()}")
                
            except Exception as e:
                self.model_loading = False
                self.model_error = str(e)
                logger.error(f"❌ Failed to load model: {e}")
        
        # Start loading in background thread
        thread = threading.Thread(target=load_model, daemon=True)
        thread.start()
    
    def run(self, debug=False):
        """Start the web server"""
        logger.info(f"🚀 Starting CometAI Web API Server...")
        logger.info(f"📡 Server will be available at: http://{self.host}:{self.port}")
        logger.info(f"🤖 Model: {self.model_name}")
        
        if debug:
            logger.info("🔧 Debug mode enabled")
        
        try:
            self.app.run(
                host=self.host,
                port=self.port,
                debug=debug,
                threaded=True,
                use_reloader=False  # Disable reloader to prevent model loading twice
            )
        except KeyboardInterrupt:
            logger.info("👋 Server stopped by user")
        except Exception as e:
            logger.error(f"❌ Server error: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CometAI Web API Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--model", default="qwen2.5-coder-7b-instruct", help="Model to use")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    try:
        server = CometAIWebServer(
            host=args.host,
            port=args.port,
            model_name=args.model,
            config_path=args.config
        )
        server.run(debug=args.debug)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
