#!/usr/bin/env python3
"""
LocalLLM GUI Chat Interface

ChatGPT-style interface with orange, black, and white theme.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime

# Add localllm to path
sys.path.append(str(Path(__file__).parent))

try:
    from localllm import LocalLLM
except ImportError as e:
    print(f"Error importing LocalLLM: {e}")
    print("Make sure all required dependencies are installed.")
    sys.exit(1)

# Modern Color Scheme - Refined Orange, Black, White
COLORS = {
    'bg_primary': '#0f0f0f',      # Deep dark background
    'bg_secondary': '#1a1a1a',    # Card backgrounds
    'bg_tertiary': '#252525',     # Input areas
    'bg_hover': '#2a2a2a',        # Hover states
    'text_primary': '#ffffff',    # Pure white text
    'text_secondary': '#b0b0b0',  # Muted gray text
    'text_tertiary': '#808080',   # Subtle gray text
    'accent_orange': '#ff6b35',   # Primary orange
    'accent_orange_hover': '#ff7a47',  # Orange hover
    'accent_orange_light': '#ff8c5a',  # Light orange
    'accent_orange_dark': '#e55a2b',   # Dark orange
    'user_bubble': '#ff6b35',     # User message bubble
    'ai_bubble': '#1e1e1e',       # AI message bubble
    'border': '#333333',          # Subtle borders
    'border_light': '#404040',    # Light borders
    'success': '#00d084',         # Modern green
    'error': '#ff4757',           # Modern red
    'warning': '#ffa502',         # Modern orange
    'shadow': '#000000',          # Shadow color
}

class ChatMessage:
    """Represents a chat message"""
    def __init__(self, role: str, content: str, timestamp: Optional[str] = None):
        self.role = role  # 'user' or 'assistant'
        self.content = content
        self.timestamp = timestamp or datetime.now().strftime("%H:%M")

class LocalLLMGUI:
    """Main GUI application"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.llm = None
        self.conversation_history = []
        self.is_generating = False
        
        # Setup window
        self.setup_window()
        
        # Setup styles
        self.setup_styles()
        
        # Create GUI elements
        self.create_widgets()
        
        # Setup logging
        self.setup_logging()
        
        # Load settings
        self.load_settings()
        
        # Initialize model in background
        self.initialize_model()
    
    def setup_window(self):
        """Setup main window with modern styling"""
        self.root.title("☄️ CometAI - AI Chat Assistant")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)
        
        # Modern window styling
        try:
            # macOS specific styling
            self.root.tk.call('tk', 'scaling', 1.2)  # Better DPI scaling
        except:
            pass
        
        # Configure window background with gradient effect
        self.root.configure(bg=COLORS['bg_primary'])
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_styles(self):
        """Setup custom styles"""
        self.style = ttk.Style()
        
        # Configure ttk styles
        self.style.theme_use('clam')
        
        # Modern Button styles
        self.style.configure(
            'Orange.TButton',
            background=COLORS['accent_orange'],
            foreground=COLORS['text_primary'],
            borderwidth=0,
            focuscolor='none',
            font=('SF Pro Display', 11, 'bold'),
            padding=(20, 12),
            relief='flat'
        )
        
        self.style.map(
            'Orange.TButton',
            background=[('active', COLORS['accent_orange_hover']),
                       ('pressed', COLORS['accent_orange_dark'])]
        )
        
        # Secondary button style
        self.style.configure(
            'Secondary.TButton',
            background=COLORS['bg_tertiary'],
            foreground=COLORS['text_primary'],
            borderwidth=1,
            focuscolor='none',
            font=('SF Pro Display', 10),
            padding=(16, 10),
            relief='flat'
        )
        
        self.style.map(
            'Secondary.TButton',
            background=[('active', COLORS['bg_hover']),
                       ('pressed', COLORS['bg_tertiary'])]
        )
        
        # Modern Entry styles
        self.style.configure(
            'Dark.TEntry',
            fieldbackground=COLORS['bg_tertiary'],
            background=COLORS['bg_tertiary'],
            foreground=COLORS['text_primary'],
            borderwidth=0,
            insertcolor=COLORS['accent_orange'],
            font=('SF Pro Display', 11),
            padding=(16, 12)
        )
        
        # Combobox styles
        self.style.configure(
            'Dark.TCombobox',
            fieldbackground=COLORS['bg_tertiary'],
            background=COLORS['bg_tertiary'],
            foreground=COLORS['text_primary'],
            arrowcolor=COLORS['accent_orange']
        )
        
        # Frame styles
        self.style.configure(
            'Dark.TFrame',
            background=COLORS['bg_primary']
        )
        
        # Modern Label styles
        self.style.configure(
            'Dark.TLabel',
            background=COLORS['bg_primary'],
            foreground=COLORS['text_primary'],
            font=('SF Pro Display', 11)
        )
        
        self.style.configure(
            'Title.TLabel',
            background=COLORS['bg_primary'],
            foreground=COLORS['accent_orange'],
            font=('SF Pro Display', 24, 'bold')
        )
        
        self.style.configure(
            'Subtitle.TLabel',
            background=COLORS['bg_primary'],
            foreground=COLORS['text_secondary'],
            font=('SF Pro Display', 13)
        )
        
        self.style.configure(
            'Card.TLabel',
            background=COLORS['bg_secondary'],
            foreground=COLORS['text_primary'],
            font=('SF Pro Display', 10)
        )
    
    def create_widgets(self):
        """Create all GUI widgets with modern styling"""
        # Main container with padding
        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=24, pady=24)
        
        # Create header
        self.create_header(main_frame)
        
        # Create main content area
        content_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Create sidebar
        self.create_sidebar(content_frame)
        
        # Create chat area
        self.create_chat_area(content_frame)
        
        # Create input area
        self.create_input_area(main_frame)
        
        # Create status bar
        self.create_status_bar(main_frame)
    
    def create_header(self, parent):
        """Create modern header with title and controls"""
        # Header container with background
        header_container = tk.Frame(parent, bg=COLORS['bg_secondary'], height=80)
        header_container.pack(fill=tk.X, pady=(0, 24))
        header_container.pack_propagate(False)
        
        # Add subtle shadow effect
        shadow_frame = tk.Frame(parent, bg=COLORS['shadow'], height=1)
        shadow_frame.pack(fill=tk.X, pady=(0, 2))
        
        header_frame = tk.Frame(header_container, bg=COLORS['bg_secondary'])
        header_frame.pack(fill=tk.BOTH, expand=True, padx=32, pady=20)
        
        # Left side - Title and status
        left_frame = tk.Frame(header_frame, bg=COLORS['bg_secondary'])
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Title with modern typography
        title_label = tk.Label(
            left_frame,
            text="☄️ CometAI",
            bg=COLORS['bg_secondary'],
            fg=COLORS['accent_orange'],
            font=('SF Pro Display', 28, 'bold')
        )
        title_label.pack(anchor=tk.W)
        
        # Subtitle
        subtitle_label = tk.Label(
            left_frame,
            text="Your Personal AI Coding Assistant",
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_secondary'],
            font=('SF Pro Display', 13)
        )
        subtitle_label.pack(anchor=tk.W, pady=(2, 0))
        
        # Right side - Controls and status
        right_frame = tk.Frame(header_frame, bg=COLORS['bg_secondary'])
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Model status with modern styling
        status_frame = tk.Frame(right_frame, bg=COLORS['bg_secondary'])
        status_frame.pack(side=tk.TOP, anchor=tk.E, pady=(0, 12))
        
        self.model_status_label = tk.Label(
            status_frame,
            text="🔄 Loading model...",
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_secondary'],
            font=('SF Pro Display', 11)
        )
        self.model_status_label.pack()
        
        # Controls frame
        controls_frame = tk.Frame(right_frame, bg=COLORS['bg_secondary'])
        controls_frame.pack(side=tk.TOP, anchor=tk.E)
        
        # Modern buttons with rounded appearance
        self.create_modern_button(controls_frame, "New Chat", self.new_chat, 'primary')
        self.create_modern_button(controls_frame, "Settings", self.show_settings, 'secondary')
    
    def create_modern_button(self, parent, text, command, style='primary'):
        """Create a modern button with rounded corners effect"""
        if style == 'primary':
            bg_color = COLORS['accent_orange']
            hover_color = COLORS['accent_orange_hover']
            text_color = COLORS['text_primary']
        else:
            bg_color = COLORS['bg_tertiary']
            hover_color = COLORS['bg_hover']
            text_color = COLORS['text_primary']
        
        button = tk.Button(
            parent,
            text=text,
            bg=bg_color,
            fg=text_color,
            font=('SF Pro Display', 11, 'bold'),
            border=0,
            padx=24,
            pady=12,
            cursor='hand2',
            command=command,
            relief='flat'
        )
        
        # Hover effects
        def on_enter(e):
            button.config(bg=hover_color)
        
        def on_leave(e):
            button.config(bg=bg_color)
        
        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)
        
        button.pack(side=tk.RIGHT, padx=(12, 0))
        return button
    
    def create_sidebar(self, parent):
        """Create modern sidebar with model selection and info"""
        # Sidebar container with rounded corners effect
        sidebar_container = tk.Frame(parent, bg=COLORS['bg_primary'])
        sidebar_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 24))
        
        sidebar_frame = tk.Frame(sidebar_container, bg=COLORS['bg_secondary'], width=320)
        sidebar_frame.pack(fill=tk.BOTH, expand=True)
        sidebar_frame.pack_propagate(False)
        
        # Sidebar title
        sidebar_title = tk.Label(
            sidebar_frame,
            text="Model & Settings",
            bg=COLORS['bg_secondary'],
            fg=COLORS['accent_orange'],
            font=('Segoe UI', 12, 'bold')
        )
        sidebar_title.pack(pady=(15, 10))
        
        # Model selection
        model_label = tk.Label(
            sidebar_frame,
            text="Model:",
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_primary'],
            font=('Segoe UI', 10)
        )
        model_label.pack(anchor=tk.W, padx=15, pady=(10, 5))
        
        self.model_var = tk.StringVar(value="qwen2.5-coder-7b-instruct")
        self.model_combo = ttk.Combobox(
            sidebar_frame,
            textvariable=self.model_var,
            values=["qwen2.5-coder-7b-instruct", "starcoder2-7b"],
            state="readonly",
            style='Dark.TCombobox',
            width=25
        )
        self.model_combo.pack(padx=15, pady=(0, 10))
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Generation settings
        settings_label = tk.Label(
            sidebar_frame,
            text="Generation Settings:",
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_primary'],
            font=('Segoe UI', 10, 'bold')
        )
        settings_label.pack(anchor=tk.W, padx=15, pady=(15, 5))
        
        # Temperature
        temp_label = tk.Label(
            sidebar_frame,
            text="Temperature:",
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_primary'],
            font=('Segoe UI', 9)
        )
        temp_label.pack(anchor=tk.W, padx=15, pady=(5, 2))
        
        self.temp_var = tk.DoubleVar(value=0.7)
        temp_scale = tk.Scale(
            sidebar_frame,
            from_=0.1, to=2.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.temp_var,
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_primary'],
            activebackground=COLORS['accent_orange'],
            highlightthickness=0,
            length=200
        )
        temp_scale.pack(padx=15, pady=(0, 5))
        
        # Max tokens
        tokens_label = tk.Label(
            sidebar_frame,
            text="Max Tokens:",
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_primary'],
            font=('Segoe UI', 9)
        )
        tokens_label.pack(anchor=tk.W, padx=15, pady=(10, 2))
        
        self.tokens_var = tk.IntVar(value=512)
        tokens_scale = tk.Scale(
            sidebar_frame,
            from_=50, to=2048,
            resolution=50,
            orient=tk.HORIZONTAL,
            variable=self.tokens_var,
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_primary'],
            activebackground=COLORS['accent_orange'],
            highlightthickness=0,
            length=200
        )
        tokens_scale.pack(padx=15, pady=(0, 10))
        
        # Model info
        info_label = tk.Label(
            sidebar_frame,
            text="Model Info:",
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_primary'],
            font=('Segoe UI', 10, 'bold')
        )
        info_label.pack(anchor=tk.W, padx=15, pady=(15, 5))
        
        self.info_text = tk.Text(
            sidebar_frame,
            height=8,
            width=30,
            bg=COLORS['bg_tertiary'],
            fg=COLORS['text_secondary'],
            font=('Consolas', 8),
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.info_text.pack(padx=15, pady=(0, 15))
    
    def create_chat_area(self, parent):
        """Create main chat area"""
        chat_frame = tk.Frame(parent, bg=COLORS['bg_primary'])
        chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Chat display with custom scrollbar
        self.chat_display = tk.Text(
            chat_frame,
            bg=COLORS['bg_primary'],
            fg=COLORS['text_primary'],
            font=('Segoe UI', 11),
            wrap=tk.WORD,
            state=tk.DISABLED,
            padx=20,
            pady=20,
            spacing1=5,
            spacing2=2,
            spacing3=5
        )
        
        # Custom scrollbar
        scrollbar = tk.Scrollbar(
            chat_frame,
            command=self.chat_display.yview,
            bg=COLORS['bg_secondary'],
            activebackground=COLORS['accent_orange'],
            troughcolor=COLORS['bg_tertiary']
        )
        
        self.chat_display.configure(yscrollcommand=scrollbar.set)
        
        # Pack chat display and scrollbar
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure text tags for message styling
        self.setup_chat_tags()
        
        # Add welcome message
        self.add_welcome_message()
    
    def setup_chat_tags(self):
        """Setup text tags for chat styling"""
        # User message tag
        self.chat_display.tag_configure(
            "user_msg",
            background=COLORS['user_bubble'],
            foreground=COLORS['text_primary'],
            font=('Segoe UI', 11),
            lmargin1=50,
            lmargin2=50,
            rmargin=20,
            spacing1=10,
            spacing3=10
        )
        
        # AI message tag
        self.chat_display.tag_configure(
            "ai_msg",
            background=COLORS['ai_bubble'],
            foreground=COLORS['text_primary'],
            font=('Segoe UI', 11),
            lmargin1=20,
            lmargin2=20,
            rmargin=50,
            spacing1=10,
            spacing3=10
        )
        
        # Timestamp tag
        self.chat_display.tag_configure(
            "timestamp",
            foreground=COLORS['text_secondary'],
            font=('Segoe UI', 9),
            justify=tk.RIGHT
        )
        
        # Code tag
        self.chat_display.tag_configure(
            "code",
            background=COLORS['bg_tertiary'],
            foreground=COLORS['accent_orange_light'],
            font=('Consolas', 10)
        )
    
    def create_input_area(self, parent):
        """Create message input area"""
        input_frame = tk.Frame(parent, bg=COLORS['bg_secondary'], height=100)
        input_frame.pack(fill=tk.X, pady=(10, 0))
        input_frame.pack_propagate(False)
        
        # Input text area
        input_container = tk.Frame(input_frame, bg=COLORS['bg_secondary'])
        input_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        self.input_text = tk.Text(
            input_container,
            height=3,
            bg=COLORS['bg_tertiary'],
            fg=COLORS['text_primary'],
            font=('Segoe UI', 11),
            wrap=tk.WORD,
            padx=15,
            pady=10,
            insertbackground=COLORS['text_primary']
        )
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Send button
        self.send_button = tk.Button(
            input_container,
            text="Send",
            bg=COLORS['accent_orange'],
            fg=COLORS['text_primary'],
            font=('Segoe UI', 10, 'bold'),
            border=0,
            padx=20,
            pady=10,
            command=self.send_message,
            cursor='hand2'
        )
        self.send_button.pack(side=tk.RIGHT, padx=(10, 0), fill=tk.Y)
        
        # Bind Enter key
        self.input_text.bind('<Return>', self.on_enter_key)
        self.input_text.bind('<Shift-Return>', self.on_shift_enter)
        
        # Focus on input
        self.input_text.focus_set()
    
    def create_status_bar(self, parent):
        """Create status bar"""
        status_frame = tk.Frame(parent, bg=COLORS['bg_secondary'], height=25)
        status_frame.pack(fill=tk.X, pady=(5, 0))
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            text="Ready",
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_secondary'],
            font=('Segoe UI', 9),
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, padx=10, pady=2)
        
        # Token counter
        self.token_label = tk.Label(
            status_frame,
            text="Tokens: 0",
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_secondary'],
            font=('Segoe UI', 9)
        )
        self.token_label.pack(side=tk.RIGHT, padx=10, pady=2)
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_settings(self):
        """Load settings from config file"""
        try:
            config_path = Path("config.yaml")
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Load model settings
                model_config = config.get('model', {})
                self.model_var.set(model_config.get('name', 'qwen2.5-coder-7b-instruct'))
                self.temp_var.set(model_config.get('temperature', 0.7))
                self.tokens_var.set(model_config.get('max_tokens', 512))
                
        except Exception as e:
            self.logger.warning(f"Could not load settings: {e}")
    
    def initialize_model(self):
        """Initialize the LLM model in background"""
        def init_model():
            try:
                self.update_status("Loading model...")
                
                # Check if transformers is available
                try:
                    import transformers
                except ImportError:
                    raise ImportError("transformers library not found. Install with: pip install transformers")
                
                self.llm = LocalLLM(model_name=self.model_var.get())
                
                # Update model info
                self.update_model_info()
                
                self.update_status("Model loaded successfully")
                self.model_status_label.config(
                    text=f"✅ {self.model_var.get()}",
                    foreground=COLORS['success']
                )
                
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                self.update_status(f"Error: {str(e)}")
                self.model_status_label.config(
                    text="❌ Model failed to load",
                    foreground=COLORS['error']
                )
                
                # Show error dialog with helpful message
                error_msg = str(e)
                if "transformers" in error_msg.lower():
                    error_msg += "\n\nInstall missing dependencies:\npip install transformers torch"
                elif "no module named" in error_msg.lower():
                    error_msg += "\n\nDownload a model first:\npython model_downloader.py download qwen2.5-coder-7b-instruct"
                
                messagebox.showerror(
                    "Model Error",
                    f"Failed to load model:\n{error_msg}"
                )
        
        # Start in background thread
        thread = threading.Thread(target=init_model, daemon=True)
        thread.start()
    
    def update_model_info(self):
        """Update model information display"""
        if not self.llm:
            return
        
        try:
            info = self.llm.get_model_info()
            
            info_text = f"""Model: {info['model_name']}
Description: {info['description']}
Parameters: {info['total_parameters']:,}
Device: {info['device']}
Quantization: {info['quantization']}
Max Tokens: {info['max_tokens']}"""
            
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info_text)
            self.info_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
    
    def add_welcome_message(self):
        """Add welcome message to chat"""
        welcome_msg = """Welcome to CometAI! 🤖

I'm your local AI coding assistant. I can help you with:
• Code generation and debugging
• Explaining complex concepts
• Code reviews and optimization
• General programming questions

Your conversations are completely private and run locally on your machine.

Copyright (c) 2025 Joshua Terranova. All rights reserved.

How can I help you today?"""
        
        self.add_message_to_chat("assistant", welcome_msg)
    
    def add_message_to_chat(self, role: str, content: str, timestamp: Optional[str] = None):
        """Add a message to the chat display"""
        if not timestamp:
            timestamp = datetime.now().strftime("%H:%M")
        
        self.chat_display.config(state=tk.NORMAL)
        
        # Add some spacing
        if self.chat_display.get(1.0, tk.END).strip():
            self.chat_display.insert(tk.END, "\n\n")
        
        # Add timestamp
        self.chat_display.insert(tk.END, f"{timestamp}\n", "timestamp")
        
        # Add message with appropriate styling
        tag = "user_msg" if role == "user" else "ai_msg"
        prefix = "You: " if role == "user" else "CometAI: "
        
        self.chat_display.insert(tk.END, f"{prefix}{content}\n", tag)
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def send_message(self):
        """Send user message and get AI response"""
        if self.is_generating:
            return
        
        message = self.input_text.get(1.0, tk.END).strip()
        if not message:
            return
        
        if not self.llm:
            messagebox.showwarning("Model Not Ready", "Please wait for the model to load.")
            return
        
        # Clear input
        self.input_text.delete(1.0, tk.END)
        
        # Add user message to chat
        self.add_message_to_chat("user", message)
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Generate AI response in background
        self.generate_response(message)
    
    def generate_response(self, message: str):
        """Generate AI response in background thread"""
        def generate():
            try:
                self.is_generating = True
                self.update_status("Generating response...")
                self.send_button.config(state=tk.DISABLED, text="Generating...")
                
                # Generate response
                response = self.llm.chat(
                    message,
                    conversation_history=self.conversation_history,
                    max_tokens=self.tokens_var.get(),
                    temperature=self.temp_var.get()
                )
                
                # Add AI response to chat
                self.root.after(0, lambda: self.add_message_to_chat("assistant", response))
                
                # Add to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Update token count (approximate)
                token_count = len(response.split()) * 1.3  # Rough estimate
                self.root.after(0, lambda: self.token_label.config(text=f"Tokens: ~{int(token_count)}"))
                
                self.root.after(0, lambda: self.update_status("Response generated"))
                
            except Exception as e:
                self.logger.error(f"Generation error: {e}")
                self.root.after(0, lambda: self.add_message_to_chat("assistant", f"Error: {str(e)}"))
                self.root.after(0, lambda: self.update_status(f"Error: {str(e)}"))
            
            finally:
                self.is_generating = False
                self.root.after(0, lambda: self.send_button.config(state=tk.NORMAL, text="Send"))
        
        thread = threading.Thread(target=generate, daemon=True)
        thread.start()
    
    def update_status(self, message: str):
        """Update status bar message"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def on_enter_key(self, event):
        """Handle Enter key press"""
        self.send_message()
        return "break"  # Prevent default behavior
    
    def on_shift_enter(self, event):
        """Handle Shift+Enter (new line)"""
        return None  # Allow default behavior (new line)
    
    def on_model_change(self, event):
        """Handle model selection change"""
        new_model = self.model_var.get()
        
        if messagebox.askyesno(
            "Change Model",
            f"Switch to {new_model}?\n\nThis will reload the model and may take a moment."
        ):
            self.model_status_label.config(
                text="Loading new model...",
                foreground=COLORS['warning']
            )
            
            # Reload model in background
            def reload_model():
                try:
                    self.llm = LocalLLM(model_name=new_model)
                    self.root.after(0, self.update_model_info)
                    self.root.after(0, lambda: self.model_status_label.config(
                        text=f"✅ {new_model}",
                        foreground=COLORS['success']
                    ))
                    self.root.after(0, lambda: self.update_status("Model switched successfully"))
                    
                except Exception as e:
                    self.logger.error(f"Failed to switch model: {e}")
                    self.root.after(0, lambda: self.model_status_label.config(
                        text="❌ Model switch failed",
                        foreground=COLORS['error']
                    ))
                    self.root.after(0, lambda: self.update_status(f"Error: {str(e)}"))
            
            thread = threading.Thread(target=reload_model, daemon=True)
            thread.start()
    
    def new_chat(self):
        """Start a new chat conversation"""
        if messagebox.askyesno("New Chat", "Start a new conversation?\n\nThis will clear the current chat."):
            self.conversation_history = []
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete(1.0, tk.END)
            self.chat_display.config(state=tk.DISABLED)
            self.add_welcome_message()
            self.update_status("New chat started")
    
    def show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.configure(bg=COLORS['bg_primary'])
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Settings content
        settings_label = tk.Label(
            settings_window,
            text="Settings",
            bg=COLORS['bg_primary'],
            fg=COLORS['accent_orange'],
            font=('Segoe UI', 14, 'bold')
        )
        settings_label.pack(pady=20)
        
        # Placeholder for settings
        info_label = tk.Label(
            settings_window,
            text="Settings panel coming soon!\n\nFor now, you can adjust generation settings\nin the sidebar.",
            bg=COLORS['bg_primary'],
            fg=COLORS['text_primary'],
            font=('Segoe UI', 10),
            justify=tk.CENTER
        )
        info_label.pack(pady=20)
        
        # Close button
        close_btn = tk.Button(
            settings_window,
            text="Close",
            bg=COLORS['accent_orange'],
            fg=COLORS['text_primary'],
            font=('Segoe UI', 10, 'bold'),
            border=0,
            padx=20,
            pady=5,
            command=settings_window.destroy
        )
        close_btn.pack(pady=20)
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_generating:
            if not messagebox.askyesno("Exit", "AI is currently generating a response.\n\nExit anyway?"):
                return
        
        # Save conversation history
        try:
            history_file = Path("gui_chat_history.json")
            with open(history_file, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save chat history: {e}")
        
        self.root.destroy()
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

def main():
    """Main function"""
    try:
        app = LocalLLMGUI()
        app.run()
    except Exception as e:
        print(f"Failed to start GUI: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
