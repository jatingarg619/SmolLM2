# SmolLM2 Model Architecture

SmolLM2 is a compact language model based on the LLaMA architecture. Here's a detailed breakdown of its architecture and parameters.

## Model Configuration

```python
vocab_size = 49152
hidden_size = 576
intermediate_size = 1536
num_hidden_layers = 30
num_attention_heads = 9
num_key_value_heads = 3
max_position_embeddings = 2048
```

## Parameter Calculation

### 1. Token Embeddings
- Input Embeddings: vocab_size × hidden_size = 49,152 × 576 = 28,311,552
- Position Embeddings: Not used (using RoPE instead)

### 2. Transformer Layers (×30)
Each layer contains:

#### Attention Block
- Q Projection: hidden_size × hidden_size = 576 × 576 = 331,776
- K Projection: hidden_size × hidden_size = 576 × 576 = 331,776
- V Projection: hidden_size × hidden_size = 576 × 576 = 331,776
- Output Projection: hidden_size × hidden_size = 576 × 576 = 331,776
- Layer Norm Weights: hidden_size = 576

#### MLP Block
- Gate Projection: hidden_size × intermediate_size = 576 × 1,536 = 884,736
- Up Projection: hidden_size × intermediate_size = 576 × 1,536 = 884,736
- Down Projection: intermediate_size × hidden_size = 1,536 × 576 = 884,736
- Layer Norm Weights: hidden_size = 576

Total per layer: 3,982,464 parameters

### 3. Final Layer
- Output Layer Norm: hidden_size = 576
- LM Head (tied with input embeddings): 0 additional parameters

### Total Parameters
1. Embeddings: 28,311,552
2. Transformer Layers: 30 × 3,982,464 = 119,473,920
3. Final Layer Norm: 576

**Total: ~147.8M parameters**

## Architecture Features

1. **RMSNorm**: Used instead of LayerNorm for better stability and efficiency
2. **Rotary Position Embeddings (RoPE)**: For handling positional information
3. **Grouped Query Attention**: Using num_key_value_heads=3 for efficiency
4. **SwiGLU Activation**: In the MLP blocks for better performance
5. **Weight Tying**: Between input embeddings and output layer

## Memory Footprint
- FP16 Model Size: ~74MB
- Activation Memory (batch_size=1, seq_len=2048): ~24MB

## Training Configuration
- Maximum Sequence Length: 2048 tokens
- Vocabulary Size: 49,152 tokens
- Learning Rate: Cosine schedule with warmup
- Optimizer: AdamW with weight decay
