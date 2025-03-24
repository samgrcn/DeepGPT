import os
import time
import math
import torch
import tiktoken
from torch.nn import functional as F
from torch.utils.data import Dataset
from model import GPT
from config import GPTConfig
from tqdm import tqdm

# Enable TensorFloat32 for better performance on Ampere and newer GPUs
torch.set_float32_matmul_precision('high')

def get_lr(it, config):
    # 1) linear warmup for warmup_iters steps
    if it < config.warmup_tokens:
        return config.learning_rate * it / config.warmup_tokens
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config.final_tokens:
        return config.learning_rate * 0.1
    # 3) in between, use cosine decay down to 10% of original
    decay_ratio = (it - config.warmup_tokens) / (config.final_tokens - config.warmup_tokens)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.learning_rate * 0.1 + coeff * (config.learning_rate - config.learning_rate * 0.1)

class TextDataset(Dataset):
    def __init__(self, data, block_size, tokenizer):
        # Tokenize the entire text
        data_tokens = tokenizer.encode(data)
        print(f"Total tokens in dataset: {len(data_tokens)}")
        self.block_size = block_size
        self.tokens = torch.tensor(data_tokens, dtype=torch.long)
    
    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def main():
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # get tiny shakespeare dataset
    with open('input.txt', 'r') as f:
        text = f.read()
    print(f"Dataset length: {len(text)} characters")
    
    # init config and model
    config = GPTConfig(model_type='small')  # use 'small' instead of 'medium'
    config.vocab_size = tokenizer.n_vocab
    model = GPT(config)
    
    # prepare the data
    train_dataset = TextDataset(text, config.block_size, tokenizer)
    print(f"Model configuration:")
    print(f"- Model type: small (124M parameters)")
    print(f"- Vocab size: {config.vocab_size}")
    print(f"- Block size: {config.block_size}")
    print(f"- Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"- Batch size: 64")
    print(f"- Learning rate: {config.learning_rate}")
    print(f"- Training tokens: {config.final_tokens}")
    print(f"- Using flash attention: {config.use_flash_attention}")
    print(f"- Using bfloat16: {config.use_bfloat16}")
    print(f"- Using torch.compile: {config.use_compile}")
    
    # setup training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    model.to(device)
    
    # Compile model if requested and available
    if config.use_compile and device == 'cuda':
        print("Compiling model...")
        # Use 'reduce-overhead' mode to better handle complex operations
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    
    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )
    
    # training loop
    batch_size = 64
    best_loss = float('inf')
    tokens = 0
    start_time = time.time()
    
    # Create progress bar
    pbar = tqdm(total=config.final_tokens, desc="Training")
    last_tokens = 0
    
    # Set up automatic mixed precision (bfloat16)
    # Use torch.amp.autocast directly instead of deprecated GradScaler
    amp_enabled = (device == 'cuda' and config.use_bfloat16)
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
    # No scaler when using autocast directly
    scaler = None
    
    model.train()
    while True:
        # get a batch of data
        ix = torch.randint(len(train_dataset), (batch_size,))
        x, y = zip(*[train_dataset[i] for i in ix])
        x = torch.stack(x).to(device)
        y = torch.stack(y).to(device)
        
        # forward pass with mixed precision if enabled
        if amp_enabled:
            with torch.autocast(device_type=device, dtype=amp_dtype):
                logits, loss = model(x, y)
            # regular backward pass without scaler
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        else:
            # regular forward and backward pass
            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        
        optimizer.zero_grad(set_to_none=True)
        
        # update learning rate
        tokens += batch_size * config.block_size
        lr = get_lr(tokens, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # update progress bar
        pbar.update(tokens - last_tokens)
        last_tokens = tokens
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.2e}'})
        
        # Generate sample occasionally
        if tokens % 10000 == 0:
            model.eval()
            with torch.no_grad():
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                print(f"\nSample at {tokens} tokens:")
                print("-"*40)
                for _ in range(100):
                    with torch.autocast(device_type=device, dtype=amp_dtype) if amp_enabled else torch.no_grad():
                        logits, _ = model(context)
                    probs = torch.softmax(logits[:, -1, :], dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    context = torch.cat([context, next_token], dim=1)
                    if context.size(1) > config.block_size:
                        context = context[:, -config.block_size:]
                generated_text = tokenizer.decode(context[0].tolist())
                print(generated_text)
                print("-"*40)
            model.train()
        
        # save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'best_model.pt')
        
        # stop if we've processed enough tokens
        if tokens > config.final_tokens:
            break
    
    pbar.close()
    print(f"\nTraining finished! Total time: {time.time() - start_time:.2f}s")
    print(f"Best loss: {best_loss:.4f}")

if __name__ == '__main__':
    main() 