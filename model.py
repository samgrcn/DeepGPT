import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.drop_p = config.dropout
        
        # flash attention flag
        self.use_flash_attention = config.use_flash_attention
        
        # Multi-head latent attention
        self.use_latent_attention = getattr(config, 'use_latent_attention', False)
        self.latent_size = getattr(config, 'latent_size', 64)  # Default latent size
        
        if self.use_latent_attention:
            # Latent projections for keys and values
            head_dim = self.n_embd // self.n_head
            self.latent_k = nn.Parameter(torch.empty(self.n_head, self.latent_size, head_dim))
            self.latent_v = nn.Parameter(torch.empty(self.n_head, self.latent_size, head_dim))
            # Initialize latent parameters
            nn.init.normal_(self.latent_k, mean=0.0, std=0.02)
            nn.init.normal_(self.latent_v, mean=0.0, std=0.02)
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        # calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # move head forward to be the batch dim
        head_dim = C // self.n_head
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hs)

        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # flash attention - much more memory efficient
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.drop_p if self.training else 0, is_causal=True)
        elif self.use_latent_attention:
            # multi-head latent attention
            # Project queries to latent space: (B, nh, T, d) x (nh, L, d) -> (B, nh, T, L)
            q_latent = torch.einsum('bnti,nli->bntl', q, self.latent_k)
            q_latent = q_latent / math.sqrt(head_dim)
            
            # causal mask to attention scores
            causal_mask = torch.triu(torch.ones(T, self.latent_size, device=x.device), diagonal=1).bool()
            q_latent = q_latent.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # softmax to get attention weights
            attn_weights = F.softmax(q_latent, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            # Apply attention to latent values
            y = torch.einsum('bntl,nlj->bntj', attn_weights, self.latent_v)
        else:
            # manual attention with dropout
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool(), float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # weight tying
        self.transformer.wte.weight = self.lm_head.weight
        
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print(f"Number of parameters: {sum(p.numel() for p in self.parameters())/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss 