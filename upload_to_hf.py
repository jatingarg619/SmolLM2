import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
from model import SmolLM2, SmolLMConfig
import torch
from pathlib import Path
import json

def get_latest_checkpoint():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(current_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        raise ValueError("No checkpoints directory found!")
        
    checkpoints = list(Path(checkpoint_dir).glob("checkpoint_*.pt"))
    if not checkpoints:
        raise ValueError("No checkpoints found!")
        
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[1]))
    step = int(latest_checkpoint.stem.split('_')[1])
    return str(latest_checkpoint), step

def save_model_for_hf(model, config, output_dir):
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model weights
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    # Save the config
    config_dict = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "hidden_act": config.hidden_act,
        "max_position_embeddings": config.max_position_embeddings,
        "initializer_range": config.initializer_range,
        "rms_norm_eps": config.rms_norm_eps,
        "use_cache": config.use_cache,
        "pad_token_id": config.pad_token_id,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "tie_word_embeddings": config.tie_word_embeddings,
        "rope_theta": config.rope_theta,
        "architectures": ["SmolLM2"],
        "model_type": "smollm2"
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

def upload_model_to_hub():
    # Load environment variables
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file")
    
    # Login to Hugging Face
    login(hf_token)
    
    # Initialize the model and tokenizer
    config = SmolLMConfig()
    model = SmolLM2(config)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    
    # Load the latest checkpoint
    latest_checkpoint, step = get_latest_checkpoint()
    print(f"Loading checkpoint from step {step}")
    checkpoint = torch.load(latest_checkpoint, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create a temporary directory for the model files
    temp_dir = "temp_model"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save the model and tokenizer
    print("Saving model and tokenizer...")
    save_model_for_hf(model, config, temp_dir)
    tokenizer.save_pretrained(temp_dir)
    
    # Create model card
    model_card = f"""---
language:
- en
tags:
- pytorch
- causal-lm
- text-generation
license: mit
---

# SmolLM2 Fine-tuned Model

This is a fine-tuned version of SmolLM2-135M. The model was trained for {step} steps.

## Model Details
- Base model: SmolLM2-135M
- Training steps: {step}
- Parameters: 135M
- Context length: 2048
- Trained on custom text data
"""
    
    with open(os.path.join(temp_dir, "README.md"), "w") as f:
        f.write(model_card)
    
    # Upload to Hub
    repo_id = "jatingocodeo/SmolLM2"
    print(f"Uploading to {repo_id}...")
    
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    
    # Upload all files in the temp directory
    api.upload_folder(
        folder_path=temp_dir,
        repo_id=repo_id,
        repo_type="model"
    )
    
    print("Upload completed successfully!")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print("Temporary files cleaned up")

if __name__ == "__main__":
    upload_model_to_hub() 