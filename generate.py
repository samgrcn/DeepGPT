import torch
import tiktoken
from model import GPT
from config import GPTConfig

def generate(model, tokenizer, prompt="", max_tokens=150, temperature=0.9):
    model.eval()
    block_size = model.config.block_size
    
    # Encode the prompt
    if prompt:
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
        # Truncate if prompt is too long
        if context.size(1) > block_size:
            context = context[:, -block_size:]
    else:
        context = torch.zeros((1, 1), dtype=torch.long)
    
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
    config = GPTConfig()  # small by default
    config.vocab_size = tokenizer.n_vocab
    model = GPT(config)
    
    print("Loading model...")
    print(f"Model size: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    model.load_state_dict(torch.load('best_model.pt'))
    
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