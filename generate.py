import torch
import tiktoken
from model import GPT
from config import GPTConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate(model, tokenizer, prompt="", max_tokens=150, temperature=0.9):
    model.eval()
    block_size = model.config.block_size
    device = next(model.parameters()).device  # Get model's device
    
    # Encode the prompt
    if prompt:
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
        # Truncate if prompt is too long
        if context.size(1) > block_size:
            context = context[:, -block_size:]
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Generate tokens
    print("\nGenerating with prompt:", prompt if prompt else "(no prompt)")
    print("-" * 60)
    generated_text = tokenizer.decode(context[0].tolist()) if prompt else ""
    print(generated_text, end="", flush=True)
    
    for _ in range(max_tokens):
        with torch.no_grad():
            # Take last block_size tokens as context
            context_window = context[:, -block_size:]
            logits, _ = model(context_window)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to context
            context = torch.cat([context, next_token], dim=1)
            
            # Decode and print the new token
            new_text = tokenizer.decode([next_token.item()])
            generated_text += new_text
            print(new_text, end="", flush=True)
    
    print("\n" + "-" * 60)
    return generated_text

def main():
    # Initialize tokenizer and model with the same config as training
    tokenizer = tiktoken.get_encoding("gpt2")
    config = GPTConfig(model_type='small')  # small by default
    config.vocab_size = tokenizer.n_vocab
    # Disable compilation to avoid issues
    config.use_compile = False
    model = GPT(config)
    
    model = model.to(device)
    
    print("Loading model...")
    print(f"Model size: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Load the model weights with state dict fixing
    try:
        # Load state dict and fix keys if needed
        state_dict = torch.load('best_model.pt', map_location=device)
        # Remove _orig_mod prefix if it exists (from torch.compile)
        fixed_state_dict = {}
        for k, v in state_dict.items():
            fixed_state_dict[k.replace('_orig_mod.', '')] = v
        
        model.load_state_dict(fixed_state_dict, strict=False)
        print("Model loaded successfully with fixed state dict")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to strict=False
        try:
            model.load_state_dict(torch.load('best_model.pt', map_location=device), strict=False)
            print("Model loaded with strict=False")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return
    
    # Prompts of Shakespeare
    prompts = [
        "HAMLET: To be, or not to be, that is",
        "JULIET: O Romeo, Romeo! wherefore art",
        "MACBETH: Is this a dagger which I see",
        "A MIDSUMMER NIGHT'S DREAM\nACT I\nSCENE I. Athens.",
        "If music be the food of love, play",
        "Friends, Romans, countrymen, lend me",
    ]
    
    for prompt in prompts:
        # Higher temperature (0.9) for more creative outputs
        generate(model, tokenizer, prompt, max_tokens=150, temperature=0.9)

if __name__ == '__main__':
    main() 