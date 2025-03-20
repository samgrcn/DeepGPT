# Efficient GPT-2 Implementation

An efficient implementation of GPT-2 in PyTorch with modern optimizations:
- Flash Attention support
- Mixed precision training (bfloat16)
- torch.compile support
- Efficient power-of-2 dimensions
- Memory optimizations

## Features

- Multiple model sizes supported:
  - Tiny (1M parameters) - for testing
  - Small (124M parameters) - GPT-2 small
  - Medium (350M parameters) - GPT-2 medium
  - Large (800M parameters) - Custom size
  - XL (1.5B parameters) - Custom size

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

1. Train on tiny Shakespeare dataset:
```bash
# Download dataset
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Train model
python train.py
```

2. Generate text:
```bash
python generate.py
```

## Configuration

Model and training parameters can be configured in `config.py`:
- Model size (tiny to xl)
- Context window size
- Vocabulary size
- Learning rate and schedule
- Dropout rate
- Optimization flags

## Training on Cloud GPUs

The code is optimized for GPU training and includes support for:
- Lambda Labs
- Any PyTorch-compatible GPU
- Multi-GPU training (coming soon)

## License

MIT 