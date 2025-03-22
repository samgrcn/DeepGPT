# Efficient GPT-2 Implementation

An efficient GPT-2 implementation for training in PyTorch with modern optimizations:
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
- Cloud GPU
- Any PyTorch-compatible GPU
- Multi-GPU training (coming soon)

## License

Copyright 2025 Garcin Samuel

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 