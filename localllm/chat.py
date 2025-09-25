"""
LocalLLM Chat Interface Module

Simple chat interface wrapper for the LocalLLM model.
"""

from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ChatInterface:
    """Simple chat interface wrapper"""
    
    def __init__(self, model):
        self.model = model
        self.conversation_history = []
    
    def send_message(self, message: str) -> str:
        """Send a message and get response"""
        response = self.model.chat(message, self.conversation_history)
        
        # Update conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        self.conversation_history.append({
            "role": "assistant", 
            "content": response
        })
        
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history.copy()
