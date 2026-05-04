import os
import time
import logging
import traceback
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import DictConfig

class JudgeManager:
    """
    A unified manager for handling both OpenAI API and local HuggingFace model judges.
    Automatically detects judge type from config and loads the appropriate model.
    """
    
    def __init__(self, config: DictConfig, cache_dir: Optional[str] = None):
        self.config = config
        self.cache_dir = cache_dir
        self.judge_type = self._detect_judge_type()
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the judge
        self._initialize_judge()
    
    def _detect_judge_type(self) -> str:
        """Detect whether to use OpenAI API or local HF model based on config."""
        judge_model = getattr(self.config, 'judge_model', getattr(self.config, 'gpt_model', None))
        
        if judge_model is None:
            raise ValueError("No judge model specified in config. Use 'judge_model' or 'gpt_model'.")
        
        # OpenAI models
        openai_models = [
            'gpt-4', 'gpt-4-0613', 'gpt-4-1106-preview', 'gpt-4-0125-preview',
            'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4-turbo-preview',
            'gpt-3.5-turbo', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0125'
        ]
        
        if any(openai_model in judge_model for openai_model in openai_models):
            return "openai"
        else:
            return "huggingface"
    
    def _initialize_judge(self):
        """Initialize the appropriate judge based on detected type."""
        if self.judge_type == "openai":
            self._initialize_openai()
        elif self.judge_type == "huggingface":
            self._initialize_huggingface()
        else:
            raise ValueError(f"Unknown judge type: {self.judge_type}")
    
    def _initialize_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI()  # Uses OPENAI_API_KEY environment variable
            print(f"Initialized OpenAI judge with model: {self.get_judge_model()}")
        except ImportError:
            raise ImportError("OpenAI package not found. Install with: pip install openai")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
    
    def _initialize_huggingface(self):
        """Initialize HuggingFace model and tokenizer."""
        judge_model = self.get_judge_model()
        
        try:
            print(f"Loading HuggingFace judge model: {judge_model}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                judge_model, 
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                judge_model,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.model.eval()
            print(f"Successfully loaded HuggingFace judge model on device: {self.model.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace model {judge_model}: {e}")
    
    def get_judge_model(self) -> str:
        """Get the judge model name from config."""
        return getattr(self.config, 'judge_model', getattr(self.config, 'gpt_model', 'gpt-3.5-turbo-0613'))
    
    def get_temperature(self) -> float:
        """Get the temperature from config."""
        return getattr(self.config, 'judge_temperature', getattr(self.config, 'temp_gpt', 0.0))
    
    def send_judge_evaluation(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Send evaluation request to the judge (either OpenAI or HuggingFace).
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments (model override, temperature override, etc.)
        
        Returns:
            Dictionary with 'content' key containing the response and optional usage info
        """
        if self.judge_type == "openai":
            return self._send_openai_request(messages, **kwargs)
        elif self.judge_type == "huggingface":
            return self._send_huggingface_request(messages, **kwargs)
        else:
            raise ValueError(f"Unknown judge type: {self.judge_type}")
    
    def _send_openai_request(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Send request to OpenAI API."""
        model = kwargs.get('model') or self.get_judge_model()
        temperature = kwargs.get('temperature') or self.get_temperature()
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            
            return {
                'content': response.choices[0].message.content,
                'usage': {
                    'total_tokens': response.usage.total_tokens,
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens
                }
            }
            
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            logging.error(traceback.format_exc())
            print(f"OpenAI API error: {e}")
            print("Waiting 5 min to handle it")
            time.sleep(300)
            return self._send_openai_request(messages, **kwargs)
    
    def _send_huggingface_request(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Send request to local HuggingFace model."""
        temperature = kwargs.get('temperature', self.get_temperature())
        max_new_tokens = kwargs.get('max_new_tokens', 512)
        
        try:
            # Convert messages to a single prompt string
            prompt = self._format_messages_for_hf(messages)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response (only the new tokens)
            response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            return {
                'content': response.strip(),
                'usage': {
                    'total_tokens': len(outputs[0]),
                    'prompt_tokens': inputs['input_ids'].shape[1],
                    'completion_tokens': len(response_tokens)
                }
            }
            
        except Exception as e:
            logging.error(f"HuggingFace model error: {e}")
            logging.error(traceback.format_exc())
            raise RuntimeError(f"HuggingFace model inference failed: {e}")
    
    def _format_messages_for_hf(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for HuggingFace model input."""

        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            try:
                return self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception:
                pass
        
        formatted_prompt = ""
        for message in messages:
            role = message['role']
            content = message['content']
            
            if role == 'system':
                formatted_prompt += f"System: {content}\n\n"
            elif role == 'user':
                formatted_prompt += f"User: {content}\n\n"
            elif role == 'assistant':
                formatted_prompt += f"Assistant: {content}\n\n"
        
        
        formatted_prompt += "Assistant:"
        return formatted_prompt
    
    def estimate_cost(self, total_tokens: int) -> float:
        """Estimate cost for OpenAI API usage."""
        if self.judge_type != "openai":
            return 0.0
        
        model = self.get_judge_model()
        
        cost_per_million = {
            'gpt-3.5-turbo': 1.5,
            'gpt-3.5-turbo-0613': 1.5,
            'gpt-3.5-turbo-1106': 1.0,
            'gpt-3.5-turbo-0125': 0.5,
            'gpt-4': 30.0,
            'gpt-4-0613': 30.0,
            'gpt-4-1106-preview': 10.0,
            'gpt-4-0125-preview': 10.0,
            'gpt-4o': 5.0,
            'gpt-4o-mini': 0.15,
            'gpt-4-turbo': 10.0,
            'gpt-4-turbo-preview': 10.0,
        }
        
        cost = cost_per_million.get(model, 1.5)
        return (total_tokens / 1_000_000) * cost
    
    def __repr__(self):
        return f"JudgeManager(type={self.judge_type}, model={self.get_judge_model()})"