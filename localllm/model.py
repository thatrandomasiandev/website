"""
LocalLLM Model Implementation

Integrates pre-trained models (Qwen2.5-Coder-7B or StarCoder2-7B) for local inference.
Optimized for coding assistance and general chat.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import os

# Transformers imports with fallback
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        BitsAndBytesConfig, GenerationConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class LocalLLMConfig:
    """Configuration for LocalLLM model"""
    
    def __init__(self, **kwargs):
        # Model selection
        self.model_name = kwargs.get('model_name', 'qwen2.5-coder-7b-instruct')
        self.model_path = kwargs.get('model_path', None)
        
        # Available models
        self.available_models = {
            'qwen2.5-coder-7b-instruct': {
                'hf_name': 'Qwen/Qwen2.5-Coder-7B-Instruct',
                'description': 'Qwen2.5 Coder 7B - Excellent for coding tasks',
                'size': '7B parameters',
                'memory_req': '14GB+ RAM',
                'speciality': 'coding'
            },
            'starcoder2-7b': {
                'hf_name': 'bigcode/starcoder2-7b',
                'description': 'StarCoder2 7B - Advanced code generation',
                'size': '7B parameters', 
                'memory_req': '14GB+ RAM',
                'speciality': 'coding'
            }
        }
        
        # Generation settings
        self.max_tokens = kwargs.get('max_tokens', 2048)
        self.temperature = kwargs.get('temperature', 0.7)
        self.top_p = kwargs.get('top_p', 0.9)
        self.top_k = kwargs.get('top_k', 50)
        self.do_sample = kwargs.get('do_sample', True)
        self.repetition_penalty = kwargs.get('repetition_penalty', 1.1)
        
        # Performance settings
        self.use_gpu = kwargs.get('use_gpu', 'auto')
        self.device_map = kwargs.get('device_map', 'auto')
        self.load_in_8bit = kwargs.get('load_in_8bit', False)
        self.load_in_4bit = kwargs.get('load_in_4bit', False)
        self.torch_dtype = kwargs.get('torch_dtype', 'auto')
        self.trust_remote_code = kwargs.get('trust_remote_code', True)
        
        # Memory optimization
        self.low_cpu_mem_usage = kwargs.get('low_cpu_mem_usage', True)
        self.max_memory = kwargs.get('max_memory', None)
        
        # Chat settings
        self.system_prompt = kwargs.get('system_prompt', 
            "You are a helpful AI coding assistant. Provide clear, accurate, and helpful responses.")
        self.chat_template = kwargs.get('chat_template', 'auto')
        
    @classmethod
    def from_file(cls, config_path: str):
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        return cls(**config_dict)


class LocalLLMAttention(nn.Module):
    """Multi-head attention optimized for CPU inference"""
    
    def __init__(self, config: LocalLLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})")
        
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()
        
        # Compute Q, K, V
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.output(context)
        
        return output


class LocalLLMFeedForward(nn.Module):
    """Feed-forward network optimized for CPU"""
    
    def __init__(self, config: LocalLLMConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, hidden_states):
        hidden_states = self.linear1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class LocalLLMLayer(nn.Module):
    """Single transformer layer"""
    
    def __init__(self, config: LocalLLMConfig):
        super().__init__()
        self.attention = LocalLLMAttention(config)
        self.feed_forward = LocalLLMFeedForward(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class LocalLLMModel(nn.Module):
    """Main LocalLLM transformer model"""
    
    def __init__(self, config: LocalLLMConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            LocalLLMLayer(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size)
        
        # Output head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, attention_mask=None, position_ids=None):
        batch_size, seq_len = input_ids.size()
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = token_embeds + position_embeds
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        return logits


class LocalLLM:
    """Main LocalLLM interface for users"""
    
    def __init__(self, model_name: Optional[str] = None, model_path: Optional[str] = None, config_path: Optional[str] = None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required. Install with: pip install transformers")
        
        # Load configuration
        if config_path:
            self.config = LocalLLMConfig.from_file(config_path)
        else:
            self.config = LocalLLMConfig()
        
        # Override model name if provided
        if model_name:
            self.config.model_name = model_name
        if model_path:
            self.config.model_path = model_path
        
        self.device = self._get_device()
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
        logger.info(f"LocalLLM initialized with {self.config.model_name} on {self.device}")
        logger.info(f"Model info: {self.get_model_info()}")
    
    def _load_model_and_tokenizer(self):
        """Load the pre-trained model and tokenizer"""
        model_info = self.config.available_models.get(self.config.model_name)
        
        if not model_info:
            raise ValueError(f"Unknown model: {self.config.model_name}. Available: {list(self.config.available_models.keys())}")
        
        model_name_or_path = self.config.model_path or model_info['hf_name']
        
        logger.info(f"Loading model: {model_name_or_path}")
        logger.info(f"Description: {model_info['description']}")
        
        # Setup quantization if requested
        quantization_config = None
        if self.config.load_in_4bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("Using 4-bit quantization for memory efficiency")
            except Exception as e:
                logger.warning(f"4-bit quantization not available: {e}")
        elif self.config.load_in_8bit:
            try:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("Using 8-bit quantization for memory efficiency")
            except Exception as e:
                logger.warning(f"8-bit quantization not available: {e}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=self.config.trust_remote_code
        )
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        logger.info("Loading model weights...")
        model_kwargs = {
            'trust_remote_code': self.config.trust_remote_code,
            'low_cpu_mem_usage': self.config.low_cpu_mem_usage,
            'torch_dtype': self._get_torch_dtype(),
        }
        
        if quantization_config:
            model_kwargs['quantization_config'] = quantization_config
        
        if self.config.device_map == 'auto' and torch.cuda.is_available():
            model_kwargs['device_map'] = 'auto'
        
        if self.config.max_memory:
            model_kwargs['max_memory'] = self.config.max_memory
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if 'device_map' not in model_kwargs:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Setup generation config - optimized for speed
        self.generation_config = GenerationConfig(
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            do_sample=self.config.do_sample,
            repetition_penalty=self.config.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # Speed optimizations
            use_cache=True,
            early_stopping=True,
            num_beams=1,  # Greedy decoding for speed
        )
    
    def _get_torch_dtype(self):
        """Get appropriate torch dtype - optimized for Apple Silicon"""
        if self.config.torch_dtype == 'auto':
            # Apple Silicon prefers bfloat16, CUDA prefers float16
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.bfloat16  # Best for Apple Silicon
            elif torch.cuda.is_available():
                return torch.float16
            else:
                return torch.float32
        elif self.config.torch_dtype == 'float16':
            return torch.float16
        elif self.config.torch_dtype == 'bfloat16':
            return torch.bfloat16
        else:
            return torch.float32
    
    def _get_device(self):
        """Determine the best device to use - optimized for Apple Silicon"""
        if self.config.use_gpu == 'auto':
            # Check for Apple Silicon MPS first, then CUDA, then CPU
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            elif torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        elif self.config.use_gpu:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            elif torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device('cpu')
    
    def _create_simple_tokenizer(self):
        """Create a simple tokenizer (placeholder)"""
        # In a real implementation, you'd use a proper tokenizer like SentencePiece
        class SimpleTokenizer:
            def __init__(self):
                self.vocab = {f"<token_{i}>": i for i in range(32000)}
                self.vocab.update({
                    "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3,
                    " ": 4, "the": 5, "and": 6, "a": 7, "to": 8, "of": 9,
                    "I": 10, "you": 11, "it": 12, "is": 13, "in": 14,
                    "that": 15, "have": 16, "for": 17, "not": 18, "with": 19,
                    "he": 20, "as": 21, "his": 22, "on": 23, "be": 24,
                })
                self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            
            def encode(self, text: str) -> List[int]:
                # Simple word-based tokenization (placeholder)
                words = text.lower().split()
                tokens = [self.vocab.get(word, 1) for word in words]  # 1 = <unk>
                return [2] + tokens + [3]  # Add <bos> and <eos>
            
            def decode(self, tokens: List[int]) -> str:
                words = [self.reverse_vocab.get(token, "<unk>") for token in tokens]
                # Remove special tokens
                words = [w for w in words if not w.startswith("<")]
                return " ".join(words)
        
        return SimpleTokenizer()
    
    def load_model(self, model_path: str):
        """Load model weights from file"""
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}")
    
    def save_model(self, model_path: str):
        """Save model weights to file"""
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
    
    def generate(self, 
                prompt: str, 
                max_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                top_p: Optional[float] = None,
                top_k: Optional[int] = None) -> str:
        """Generate text from a prompt using the pre-trained model"""
        
        # Create generation config for this call
        gen_config = GenerationConfig(
            max_new_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            top_p=top_p or self.config.top_p,
            top_k=top_k or self.config.top_k,
            do_sample=self.config.do_sample,
            repetition_penalty=self.config.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Move to device if not using device_map
        if not hasattr(self.model, 'hf_device_map'):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
                use_cache=True
            )
        
        # Decode only the new tokens (exclude input) with error handling
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        
        try:
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        except UnicodeDecodeError:
            # Fallback: decode with error handling
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True, errors='ignore')
        
        return generated_text.strip()
    
    def chat(self, message: str, conversation_history: Optional[List[Dict]] = None, 
             max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        """Chat interface with conversation history using proper chat templates"""
        
        # Build conversation for chat template
        messages = []
        
        # Add system message
        messages.append({
            "role": "system",
            "content": self.config.system_prompt
        })
        
        # Add conversation history
        if conversation_history:
            for turn in conversation_history[-10:]:  # Keep last 10 turns
                messages.append(turn)
        
        # Add current message
        messages.append({
            "role": "user", 
            "content": message
        })
        
        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback to simple format
            prompt_parts = []
            for msg in messages:
                if msg['role'] == 'system':
                    prompt_parts.append(f"System: {msg['content']}")
                elif msg['role'] == 'user':
                    prompt_parts.append(f"User: {msg['content']}")
                elif msg['role'] == 'assistant':
                    prompt_parts.append(f"Assistant: {msg['content']}")
            
            prompt_parts.append("Assistant:")
            prompt = "\n".join(prompt_parts)
        
        # Generate response with provided parameters
        response = self.generate(
            prompt, 
            max_tokens=max_tokens or 512,
            temperature=temperature
        )
        
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        model_info = self.config.available_models.get(self.config.model_name, {})
        
        # Count parameters
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        except:
            total_params = "Unknown"
            trainable_params = "Unknown"
        
        return {
            "model_name": self.config.model_name,
            "description": model_info.get('description', 'Unknown'),
            "hf_name": model_info.get('hf_name', 'Unknown'),
            "speciality": model_info.get('speciality', 'general'),
            "size": model_info.get('size', 'Unknown'),
            "memory_requirement": model_info.get('memory_req', 'Unknown'),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "torch_dtype": str(self._get_torch_dtype()),
            "quantization": "4-bit" if self.config.load_in_4bit else "8-bit" if self.config.load_in_8bit else "None",
            "max_tokens": self.config.max_tokens,
            "vocab_size": getattr(self.tokenizer, 'vocab_size', 'Unknown')
        }
    
    def list_available_models(self) -> Dict[str, Dict]:
        """List all available models"""
        return self.config.available_models
    
    def switch_model(self, model_name: str):
        """Switch to a different model"""
        if model_name not in self.config.available_models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.config.available_models.keys())}")
        
        logger.info(f"Switching from {self.config.model_name} to {model_name}")
        self.config.model_name = model_name
        self._load_model_and_tokenizer()
        logger.info(f"Successfully switched to {model_name}")
    
    def optimize_for_inference(self):
        """Apply optimizations for faster inference"""
        logger.info("Applying inference optimizations...")
        
        # Enable torch.compile if available (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model)
                logger.info("✅ Applied torch.compile optimization")
        except Exception as e:
            logger.warning(f"torch.compile not available: {e}")
        
        # Set to eval mode and disable gradients
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        logger.info("✅ Inference optimizations applied")
