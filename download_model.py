from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def download_model():
    print("Downloading SmolLM2-135M model and tokenizer...")
    
    # Model ID
    model_id = "HuggingFaceTB/SmolLM2-135M"
    
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Tokenizer downloaded successfully!")
    
    # Download model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Load in float16 to save memory
        device_map="auto"  # Automatically handle device placement
    )
    print("Model downloaded successfully!")
    
    # Print model architecture
    print("\nModel Architecture:")
    print("===================")
    print(model)
    
    # Print number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,}")
    
    # Save model and tokenizer locally
    print("\nSaving model and tokenizer locally...")
    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")
    print("Model and tokenizer saved successfully!")

if __name__ == "__main__":
    download_model() 