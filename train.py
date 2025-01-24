import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os
from model import SmolLM2, SmolLMConfig
from pathlib import Path

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        # Get absolute path relative to this script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        abs_file_path = os.path.join(current_dir, file_path)
        
        with open(abs_file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # Tokenize the entire text
        self.tokens = tokenizer.encode(self.text)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.tokens) - self.max_length
        
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.max_length + 1]  # +1 for target
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def get_latest_checkpoint():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(current_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return None, 0
        
    checkpoints = list(Path(checkpoint_dir).glob("checkpoint_*.pt"))
    if not checkpoints:
        return None, 0
        
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[1]))
    step = int(latest_checkpoint.stem.split('_')[1])
    return str(latest_checkpoint), step

def save_checkpoint(model, optimizer, step, path):
    # Get absolute path for checkpoints directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(current_dir, "checkpoints")
    abs_path = os.path.join(checkpoint_dir, os.path.basename(path))
    
    # Create directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Delete previous checkpoint if it exists
    for f in Path(checkpoint_dir).glob("checkpoint_*.pt"):
        f.unlink()
    
    # Save new checkpoint
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, abs_path)
    print(f"Checkpoint saved at step {step}")

def generate_sample(model, tokenizer, prompt="First Citizen:", max_length=50):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        # Generate one token at a time
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / 1.0  # temperature=1.0
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            if next_token.item() == model.config.eos_token_id:
                break
                
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n{generated_text}\n")
    model.train()

def train(num_steps=5000, checkpoint_interval=500, learning_rate=1e-4):
    # Initialize model and tokenizer
    config = SmolLMConfig()
    model = SmolLM2(config)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    
    # Create dataset and dataloader
    dataset = TextDataset("input.txt", tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Load latest checkpoint if exists
    latest_checkpoint, initial_step = get_latest_checkpoint()
    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from step {initial_step}")
    else:
        initial_step = 0
    
    # Training loop
    step = initial_step
    data_iter = iter(dataloader)
    model.train()
    
    while step < initial_step + num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        input_ids, labels = batch
        outputs = model(input_ids)
        loss = torch.nn.functional.cross_entropy(outputs.view(-1, config.vocab_size), labels.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}, Loss: {loss.item():.4f}")
        
        if (step + 1) % checkpoint_interval == 0:
            checkpoint_path = f"checkpoint_{step + 1}.pt"
            save_checkpoint(model, optimizer, step + 1, checkpoint_path)
            generate_sample(model, tokenizer)
        
        step += 1
    
    return model, tokenizer

if __name__ == "__main__":
    print("Starting training...")
    model, tokenizer = train(num_steps=5000, checkpoint_interval=500)
    print("Training completed!") 