import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class SmolLMConfig:
    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.041666666666666664
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: int = 0
    eos_token_id: int = 0
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

def precompute_rope_frequencies(dim: int, max_position_embeddings: int, theta: float = 10000.0):
    position = torch.arange(max_position_embeddings).unsqueeze(1)  # [seq_len, 1]
    div_term = theta ** (torch.arange(0, dim, 2).float() / dim)   # [dim/2]
    freqs = position / div_term  # [seq_len, dim/2]
    return freqs

def apply_rotary_embeddings(x: torch.Tensor, freqs: torch.Tensor):
    # x shape: [batch, seq_len, heads, head_dim]
    # freqs shape: [seq_len, head_dim/2]
    x_rot = x.float()
    
    # Reshape freqs to match x's dimensions
    freqs = freqs.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim/2]
    
    # Split channels for rotation
    x1, x2 = x_rot[..., :x_rot.shape[-1]//2], x_rot[..., x_rot.shape[-1]//2:]
    
    # Apply rotary embeddings
    cos = torch.cos(freqs).to(x.device)
    sin = torch.sin(freqs).to(x.device)
    
    # Ensure broadcasting dimensions match
    cos = cos.expand_as(x1)
    sin = sin.expand_as(x1)
    
    # Rotate x1 and x2
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x2 * cos + x1 * sin
    
    # Concatenate back
    return torch.cat([x1_rot, x2_rot], dim=-1).to(x.dtype)

class LlamaAttention(nn.Module):
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Adjust projections to match head dimensions
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        
        # Initialize rotary embeddings
        self.register_buffer(
            "rope_freqs",
            precompute_rope_frequencies(
                self.head_dim,  # Use full head_dim for frequencies
                config.max_position_embeddings,
                config.rope_theta
            ),
            persistent=False
        )

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()
        
        # Project and reshape
        q = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        
        # Apply rotary embeddings
        q = apply_rotary_embeddings(q, self.rope_freqs[:seq_length])
        k = apply_rotary_embeddings(k, self.rope_freqs[:seq_length])
        
        # Repeat k/v heads if num_kv_heads < num_heads
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        
        # Scaled dot-product attention
        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, v)
        
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, -1)
        
        return self.o_proj(context)

class LlamaMLP(nn.Module):
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class SmolLM2(nn.Module):
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # Add lm_head before weight tying
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie weights if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
            
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal attention mask if none provided
        if attention_mask is None:
            attention_mask = torch.triu(torch.ones((input_ids.size(1), input_ids.size(1)), 
                                                 dtype=torch.bool, device=input_ids.device), 
                                     diagonal=1)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            attention_mask = attention_mask * -1e4
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)  # Use lm_head instead of F.linear
        
        return logits

    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50):
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Apply top-k filtering
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token = top_k_indices[0][torch.multinomial(F.softmax(top_k_logits, dim=-1)[0], 1)]
                
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
                if next_token.item() == self.config.eos_token_id:
                    break
                    
        return input_ids 