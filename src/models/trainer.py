import os
import json
import torch
import torch.nn as nn
from sentencepiece import SentencePieceProcessor

class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.n_kv_heads = config["n_kv_heads"]
        self.dim = config["dim"]
        head_dim = self.dim // self.n_heads
        
        self.wq = nn.Linear(self.dim, self.dim, bias=False)
        self.wk = nn.Linear(self.dim, head_dim * self.n_kv_heads, bias=False)
        self.wv = nn.Linear(self.dim, head_dim * self.n_kv_heads, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

class LlamaFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config["dim"], config["dim"], bias=False)
        self.w2 = nn.Linear(config["dim"], config["dim"], bias=False)
        self.w3 = nn.Linear(config["dim"], config["dim"], bias=False)

class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.tok_embeddings = nn.Embedding(config["vocab_size"], config["dim"])
        self.layers = nn.ModuleList()
        
        for _ in range(config["n_layers"]):
            self.layers.append(nn.ModuleDict({
                "attention": LlamaAttention(config),
                "feed_forward": LlamaFeedForward(config),
                "attention_norm": nn.LayerNorm(config["dim"], eps=config["norm_eps"]),
                "ffn_norm": nn.LayerNorm(config["dim"], eps=config["norm_eps"])
            }))
            
        self.norm = nn.LayerNorm(config["dim"], eps=config["norm_eps"])
        self.output = nn.Linear(config["dim"], config["vocab_size"], bias=False)

class LlamaFineTuner:
    def __init__(self):
        self.model_path = "/Users/mariobozza/.llama/checkpoints/Llama3.2-1B"
        self.tokenizer_path = os.path.join(self.model_path, "tokenizer.model")
        self.params_path = os.path.join(self.model_path, "params.json")
        self.checkpoint_path = os.path.join(self.model_path, "consolidated.00.pth")
        self.model = None
        self.tokenizer = None

    def setup(self):
        """Initialize the model and tokenizer."""
        print(f"Loading model from: {self.model_path}")
        
        try:
            # Load model parameters
            with open(self.params_path, 'r') as f:
                self.model_params = json.load(f)
            print("Loaded parameters:", self.model_params)
            
            # Initialize tokenizer
            self.tokenizer = SentencePieceProcessor()
            self.tokenizer.Load(self.tokenizer_path)
            print("Tokenizer loaded successfully")
            
            # Initialize model
            self.model = LlamaModel(self.model_params)
            
            # Load weights
            state_dict = torch.load(self.checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
            print("Model loaded successfully")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False