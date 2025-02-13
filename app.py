import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PretrainedConfig
from huggingface_hub import hf_hub_download
import json
import torch.nn as nn
import torch.nn.functional as F
import math

# Define the model architecture
class SmolLM2Config(PretrainedConfig):
    model_type = "smollm2"
    
    def __init__(
        self,
        vocab_size=49152,
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.041666666666666664,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=0,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )

# Register the model architecture
from transformers import AutoConfig
AutoConfig.register("smollm2", SmolLM2Config)

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
    def __init__(self, config: SmolLM2Config):
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
    def __init__(self, config: SmolLM2Config):
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
    def __init__(self, config: SmolLM2Config):
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

class SmolLM2ForCausalLM(PreTrainedModel):
    config_class = SmolLM2Config
    
    def __init__(self, config):
        super().__init__(config)
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

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal attention mask if none provided
        if attention_mask is None:
            # Create causal mask
            seq_length = input_ids.size(1)
            # [batch_size, 1, seq_length, seq_length]
            causal_mask = torch.triu(
                torch.ones((seq_length, seq_length), dtype=torch.bool, device=input_ids.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)
            attention_mask = torch.zeros(
                (1, 1, seq_length, seq_length),
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )
            attention_mask.masked_fill_(causal_mask, float("-inf"))
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            
        return logits if loss is None else (loss, logits)
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask", None)
        }

    def generate(
        self,
        input_ids,
        max_length=100,
        temperature=0.7,
        top_k=50,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=None,
        eos_token_id=None,
        **kwargs
    ):
        cur_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        
        if max_length < cur_len:
            max_length = cur_len
            
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        
        while cur_len < max_length:
            # Prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids)
            
            # Forward pass
            with torch.no_grad():
                outputs = self(**model_inputs)
                next_token_logits = outputs[:, -1, :]
            
            # Temperature scaling
            if temperature != 1.0 and temperature > 0:
                next_token_logits = next_token_logits / temperature
                
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample or greedy
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                next_tokens = next_tokens.unsqueeze(-1)
            
            # Append next tokens
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            cur_len = input_ids.shape[1]
            
            # Early stopping if all sequences have reached the EOS token
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.squeeze(-1).ne(eos_token_id).long()
                )
                if unfinished_sequences.max() == 0:
                    break
        
        return input_ids

# Register the model
AutoModelForCausalLM.register(SmolLM2Config, SmolLM2ForCausalLM)

# Cache for model and tokenizer
MODEL = None
TOKENIZER = None
CONFIG = None

def initialize():
    global MODEL, TOKENIZER, CONFIG
    
    if MODEL is None:
        print("Loading model and tokenizer...")
        model_id = "jatingocodeo/SmolLM2"
        
        try:
            # Download and load config
            print("Loading config...")
            config_path = hf_hub_download(repo_id=model_id, filename="config.json")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            CONFIG = SmolLM2Config(**config_dict)
            
            # Load tokenizer
            print("Loading tokenizer...")
            TOKENIZER = AutoTokenizer.from_pretrained(
                model_id,
                model_max_length=CONFIG.max_position_embeddings,
                padding_side="left",
                truncation_side="left",
                trust_remote_code=True
            )
            
            # Make sure we're using the correct special tokens
            special_tokens = {
                'bos_token': '<|endoftext|>',
                'eos_token': '<|endoftext|>',
                'unk_token': '<|endoftext|>',
                'pad_token': '<|endoftext|>'  # Using endoftext as pad token since it's not specified
            }
            TOKENIZER.add_special_tokens(special_tokens)
            
            # Load model weights
            print("Loading model...")
            weights_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
            
            # Initialize model
            MODEL = SmolLM2ForCausalLM(CONFIG)
            
            # Resize token embeddings to match tokenizer
            MODEL.resize_token_embeddings(len(TOKENIZER))
            
            # Load state dict
            state_dict = torch.load(weights_path, map_location="cpu")
            MODEL.load_state_dict(state_dict)
            
            # Move model to device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            MODEL = MODEL.to(device)
            
            print(f"Model loaded successfully on {device}")
            
        except Exception as e:
            print(f"Error initializing: {str(e)}")
            raise

def generate_text(prompt, max_length=100, temperature=0.7, top_k=50):
    # Initialize if not already done
    if MODEL is None:
        try:
            initialize()
        except Exception as e:
            return f"Failed to initialize model: {str(e)}"
    
    try:
        # Process prompt
        if not prompt.strip():
            return "Please enter a prompt."
        
        # Add BOS token if needed
        if not prompt.startswith(TOKENIZER.bos_token):
            prompt = TOKENIZER.bos_token + prompt
        
        # Encode prompt
        encoded = TOKENIZER.encode_plus(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CONFIG.max_position_embeddings
        )
        input_ids = encoded["input_ids"].to(MODEL.device)
        attention_mask = encoded["attention_mask"].to(MODEL.device)
        
        # Generate
        with torch.no_grad():
            outputs = MODEL.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=min(max_length + len(input_ids[0]), CONFIG.max_position_embeddings),
                temperature=max(0.1, min(temperature, 1.0)),  # Clamp temperature
                top_k=max(1, min(top_k, 100)),  # Clamp top_k
                do_sample=True if temperature > 0 else False,
                num_return_sequences=1,
                pad_token_id=TOKENIZER.pad_token_id,
                eos_token_id=TOKENIZER.eos_token_id,
            )
        
        # Decode and return
        generated_text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        return generated_text.strip()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error during text generation: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here...", lines=2),
        gr.Slider(minimum=10, maximum=200, value=100, step=1, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top K"),
    ],
    outputs=gr.Textbox(label="Generated Text", lines=5),
    title="SmolLM2 Text Generator",
    description="Generate text using the fine-tuned SmolLM2 model. Adjust parameters to control the generation.",
    examples=[
        ["Once upon a time", 100, 0.7, 50],
        ["The quick brown fox", 150, 0.8, 40],
    ],
    allow_flagging="never"
)

# Initialize on startup
try:
    initialize()
except Exception as e:
    print(f"Warning: Model initialization failed: {str(e)}")
    print("Model will be initialized on first request")

if __name__ == "__main__":
    iface.launch() 