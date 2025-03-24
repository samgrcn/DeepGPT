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
    
    def __init__(self, model_type='small', vocab_size=None, block_size=None):
        self.model_type = model_type
        
        # Default configuration for training
        self.use_flash_attention = True
        self.use_bfloat16 = True
        self.use_compile = True
        
        # DeepSeek optimizations
        self.use_latent_attention = True  # Enable latent attention by default
        self.latent_size = 64  # Default latent dimension (adjust based on model size)
        
        # Rotary Position Embeddings (RoPE)
        self.use_rope = True  # Enable RoPE by default
        self.rope_theta = 10000.0  # Default base frequency
        
        # Define model configs based on model size
        if model_type == 'tiny':
            self.n_layer = 6
            self.n_head = 6
            self.n_embd = 192
            self.dropout = 0.1
        elif model_type == 'small':  # GPT-2 small (124M)
            self.n_layer = 12
            self.n_head = 12
            self.n_embd = 768
            self.dropout = 0.1
            # Scale latent size with model size
            self.latent_size = 128
        elif model_type == 'medium':  # GPT-2 medium (350M)
            self.n_layer = 24
            self.n_head = 16
            self.n_embd = 1024
            self.dropout = 0.1
            # Scale latent size with model size
            self.latent_size = 192
        elif model_type == 'large':  # ~800M model
            self.n_layer = 36
            self.n_head = 20
            self.n_embd = 1280
            self.dropout = 0.1
            # Scale latent size with model size
            self.latent_size = 256
        elif model_type == 'xl':  # ~1.5B model
            self.n_layer = 48
            self.n_head = 24
            self.n_embd = 1600
            self.dropout = 0.1
            # Scale latent size with model size
            self.latent_size = 320
        
        # Override with provided values if specified
        self.vocab_size = vocab_size if vocab_size is not None else 49152  # 3 * 2^14 (more efficient than 50257)
        self.block_size = block_size if block_size is not None else 256
        
        # Optimizer params
        self.learning_rate = 5e-5  # Learning rate
        self.beta1 = 0.9          # Adam optimizer beta1
        self.beta2 = 0.95         # Adam optimizer beta2
        self.grad_clip = 1.0      # Gradient clipping threshold
        self.weight_decay = 0.1   # Weight decay regularization
        
        # Learning rate schedule
        self.warmup_tokens = 65536  # 2^16
        self.final_tokens = 2097152  # 2^21
        
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