from dataclasses import dataclass

@dataclass
class GPTConfig:
    # GPT-2 sizes with power-of-2 dimensions
    model_configs = {
        'tiny': dict(n_layer=4, n_head=8, n_embd=256),      # ~1M params for testing
        'small': dict(n_layer=12, n_head=16, n_embd=768),   # 124M params
        'medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
        'large': dict(n_layer=36, n_head=32, n_embd=1536),  # ~800M params
        'xl': dict(n_layer=48, n_head=32, n_embd=2048),     # ~1.5B params
    }
    
    def __init__(self, model_type='medium'):
        # model dimensions
        config = self.model_configs[model_type]
        self.n_layer = config['n_layer']
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        
        # other hyperparameters
        self.block_size = 256  # context window size (power of 2)
        self.vocab_size = 49152  # 48K = 3 * 2^14 (close to original 50257 but more efficient)
        self.dropout = 0.1
        
        # training
        self.learning_rate = 5e-5
        self.weight_decay = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0
        
        # token counts
        self.warmup_tokens = 2**16  # 65536
        self.final_tokens = 2**21   # ~2M tokens
        
        # optimization flags
        self.use_flash_attention = True
        self.use_bfloat16 = True
        self.use_compile = True 