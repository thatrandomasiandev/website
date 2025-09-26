#!/usr/bin/env python3
"""
CometAI Chat Interface

Simple command-line chat interface for CometAI.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict
import logging

# Add localllm to path
sys.path.append(str(Path(__file__).parent))

from localllm import LocalLLM

def setup_logging(verbose: bool = False):
    """Setup logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

def load_conversation_history(history_file: str) -> List[Dict]:
    """Load conversation history from file"""
    history_path = Path(history_file)
    if history_path.exists():
        try:
            with open(history_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []

def save_conversation_history(history: List[Dict], history_file: str):
    """Save conversation history to file"""
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    except IOError as e:
        logging.warning(f"Failed to save conversation history: {e}")

def print_banner():
    """Print chat interface banner"""
    print("""
☄️ CometAI Chat Interface
=========================

Your personal AI assistant running locally!

Commands:
  /help     - Show this help
  /clear    - Clear conversation history
  /info     - Show model information
  /save     - Save conversation
  /quit     - Exit chat

Type your message and press Enter to chat.
""")

def print_help():
    """Print help information"""
    print("""
💬 Chat Commands:
  /help     - Show this help message
  /clear    - Clear conversation history
  /info     - Show model information
  /save     - Save current conversation
  /history  - Show conversation history
  /quit     - Exit the chat

🎛️ Generation Settings:
  You can adjust these in config.yaml:
  • temperature: Controls randomness (0.1-2.0)
  • top_p: Controls diversity (0.1-1.0)
  • max_tokens: Maximum response length

🔧 Tips:
  • Be specific in your questions
  • Use context from previous messages
  • Try different phrasings if unsatisfied
  • The model learns from conversation context
""")

def print_model_info(llm: LocalLLM):
    """Print model information"""
    info = llm.get_model_info()
    print(f"""
🧠 Model Information:
  Name: {info['model_name']}
  Version: {info['version']}
  Parameters: {info['parameters']:,}
  Device: {info['device']}
  Vocabulary Size: {info['vocab_size']:,}
  Hidden Size: {info['hidden_size']}
  Layers: {info['num_layers']}
  Max Tokens: {info['max_tokens']}
""")

def format_response(response: str) -> str:
    """Format the AI response for display"""
    # Clean up the response
    response = response.strip()
    
    # Remove any remaining prompt artifacts
    if response.startswith("Assistant:"):
        response = response[10:].strip()
    
    return response

def main():
    """Main chat function"""
    parser = argparse.ArgumentParser(description="LocalLLM Chat Interface")
    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--history", type=str, default="chat_history.json", 
                       help="Conversation history file")
    parser.add_argument("--no-history", action="store_true", 
                       help="Don't load or save conversation history")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    parser.add_argument("--single", type=str, 
                       help="Ask a single question and exit")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Initialize CometAI
        print("☄️ Initializing CometAI...")
        llm = LocalLLM(model_path=args.model, config_path=args.config)
        print("✅ CometAI ready!")
        
        # Load conversation history
        conversation_history = []
        if not args.no_history:
            conversation_history = load_conversation_history(args.history)
            if conversation_history:
                print(f"📚 Loaded {len(conversation_history)} previous messages")
        
        # Single question mode
        if args.single:
            print(f"\n👤 You: {args.single}")
            response = llm.chat(args.single, conversation_history)
            response = format_response(response)
            print(f"🤖 LocalLLM: {response}")
            return
        
        # Interactive chat mode
        print_banner()
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input[1:].lower()
                    
                    if command == 'quit' or command == 'exit':
                        print("👋 Goodbye!")
                        break
                    elif command == 'help':
                        print_help()
                        continue
                    elif command == 'clear':
                        conversation_history = []
                        print("🧹 Conversation history cleared")
                        continue
                    elif command == 'info':
                        print_model_info(llm)
                        continue
                    elif command == 'save':
                        if not args.no_history:
                            save_conversation_history(conversation_history, args.history)
                            print(f"💾 Conversation saved to {args.history}")
                        else:
                            print("⚠️ History saving is disabled")
                        continue
                    elif command == 'history':
                        if conversation_history:
                            print(f"\n📚 Conversation History ({len(conversation_history)} messages):")
                            for i, msg in enumerate(conversation_history[-10:], 1):  # Show last 10
                                role = "👤" if msg['role'] == 'user' else "🤖"
                                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                                print(f"  {i}. {role} {content}")
                        else:
                            print("📚 No conversation history")
                        continue
                    else:
                        print(f"❌ Unknown command: /{command}")
                        print("Type /help for available commands")
                        continue
                
                # Generate AI response
                print("🤖 LocalLLM: ", end="", flush=True)
                
                response = llm.chat(user_input, conversation_history)
                response = format_response(response)
                
                print(response)
                
                # Update conversation history
                if not args.no_history:
                    conversation_history.append({
                        'role': 'user',
                        'content': user_input
                    })
                    conversation_history.append({
                        'role': 'assistant', 
                        'content': response
                    })
                    
                    # Keep history manageable (last 50 messages)
                    if len(conversation_history) > 50:
                        conversation_history = conversation_history[-50:]
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logging.error(f"Chat error: {e}", exc_info=True)
                continue
        
        # Save conversation history on exit
        if not args.no_history and conversation_history:
            save_conversation_history(conversation_history, args.history)
            print(f"💾 Conversation saved to {args.history}")
    
    except Exception as e:
        print(f"❌ Failed to initialize CometAI: {e}")
        logging.error(f"Initialization error: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
