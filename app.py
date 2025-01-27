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
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=0,
        tie_word_embeddings=True,
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

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        )
        
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
    def forward(self, hidden_states, attention_mask=None):
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Reshape for attention
        batch_size, seq_length, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            scores = scores + attention_mask
            
        attn_weights = F.softmax(scores, dim=-1)
        hidden_states = torch.matmul(attn_weights, v)
        
        # Reshape back
        hidden_states = hidden_states.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        hidden_states = self.o_proj(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP
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
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
            
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        hidden_states = self.embed_tokens(input_ids)
        
        # Process through layers
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

# Register the model
AutoModelForCausalLM.register(SmolLM2Config, SmolLM2ForCausalLM)

# Cache for model and tokenizer
MODEL = None
TOKENIZER = None

def initialize():
    global MODEL, TOKENIZER
    
    if MODEL is None:
        print("Loading model and tokenizer...")
        model_id = "jatingocodeo/SmolLM2"
        
        try:
            # Download and load config
            print("Loading config...")
            config_path = hf_hub_download(repo_id=model_id, filename="config.json")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = SmolLM2Config(**config_dict)
            
            # Load tokenizer
            print("Loading tokenizer...")
            TOKENIZER = AutoTokenizer.from_pretrained(model_id)
            
            # Add special tokens if needed
            special_tokens = {
                'pad_token': '[PAD]',
                'eos_token': '</s>',
                'bos_token': '<s>'
            }
            TOKENIZER.add_special_tokens(special_tokens)
            
            # Load model weights
            print("Loading model...")
            weights_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
            
            # Initialize model
            MODEL = SmolLM2ForCausalLM(config)
            
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
        input_ids = TOKENIZER.encode(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = input_ids.to(MODEL.device)
        
        # Generate
        with torch.no_grad():
            outputs = MODEL.generate(
                input_ids,
                max_length=min(max_length + len(input_ids[0]), 2048),
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