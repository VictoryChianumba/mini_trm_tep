# model_9x9.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TinyRecursiveModel(nn.Module):
    def __init__(
        self,
        vocab_size=10,       # 0-9 for 9×9 Sudoku
        context_length=81,   # 9×9 grid flattened
        hidden_size=256,     # Increase from 128
        n_layers=2,
        n_recursions=6,
        T_cycles=3,
        use_attention=True,  # Use Transformer instead of MLP-Mixer
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_recursions = n_recursions
        self.T_cycles = T_cycles
        self.use_attention = use_attention
        
        # Input embedding
        self.input_embedding = nn.Embedding(vocab_size, hidden_size)
        nn.init.normal_(self.input_embedding.weight, std=0.02)
        
        # Initial embeddings for y and z
        self.y_init = nn.Parameter(torch.randn(1, context_length, hidden_size) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, context_length, hidden_size) * 0.02)
        
        # Single tiny network
        if use_attention:
            self.network = TransformerNetwork(
                context_length=context_length,
                hidden_size=hidden_size,
                n_layers=n_layers
            )
        else:
            self.network = TinyNetwork(
                context_length=context_length,
                hidden_size=hidden_size,
                n_layers=n_layers
            )
        
        # Output head
        self.output_head = nn.Linear(hidden_size, vocab_size)
        nn.init.normal_(self.output_head.weight, std=0.02)
        
        # Q-head for halting
        self.q_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.q_head.weight, std=0.02)
        
    def forward(self, x, y=None, z=None, n_supervision_steps=16, training=True):
        """Original forward - keep for compatibility"""
        batch_size = x.shape[0]
        
        x_embedded = self.input_embedding(x)
        
        if y is None:
            y = self.y_init.expand(batch_size, -1, -1).clone()
        if z is None:
            z = self.z_init.expand(batch_size, -1, -1).clone()
        
        predictions = []
        q_values = []
        
        for step in range(n_supervision_steps):
            y, z, y_logits, q = self.deep_recursion(x_embedded, y, z)
            predictions.append(y_logits)
            q_values.append(q)
            
            if not training:
                if torch.sigmoid(q).mean() > 0.5:
                    break
        
        return predictions, q_values, y, z
    
    def forward_single_step(self, x_embedded, y, z):
        """
        Run ONE supervision step and return results.
        Used for step-by-step training to save memory.
        
        Args:
            x_embedded: [B, L, D] - already embedded input
            y: [B, L, D] - current answer state
            z: [B, L, D] - current reasoning state
            
        Returns:
            y: [B, L, D] - updated answer (WITH GRADIENTS)
            z: [B, L, D] - updated reasoning (WITH GRADIENTS)
            y_logits: [B, L, vocab_size] - predictions (WITH GRADIENTS)
            q: [B, 1] - halt prediction (WITH GRADIENTS)
        """
        # Deep recursion (T-1 without grad, 1 with grad)
        with torch.no_grad():
            for _ in range(self.T_cycles - 1):
                y, z = self.latent_recursion(x_embedded, y, z)
        
        # Last cycle with gradients
        y, z = self.latent_recursion(x_embedded, y, z)
        
        # Compute outputs (WITH gradients)
        y_logits = self.output_head(y)
        q = self.q_head(y.mean(dim=1))
        
        # DON'T detach here - we need gradients for this step
        return y, z, y_logits, q
    
    def deep_recursion(self, x, y, z):
        # T-1 cycles without gradients
        with torch.no_grad():
            for _ in range(self.T_cycles - 1):
                y, z = self.latent_recursion(x, y, z)
        
        # 1 cycle with gradients
        y, z = self.latent_recursion(x, y, z)
        
        # Compute outputs
        y_logits = self.output_head(y)
        q = self.q_head(y.mean(dim=1))
        
        return y.detach(), z.detach(), y_logits, q
    
    def latent_recursion(self, x, y, z):
        # Update z n times
        for _ in range(self.n_recursions):
            z_input = torch.cat([x, y, z], dim=-1)
            z_update = self.network(z_input)
            z = z + z_update
        
        # Update y once
        y_input = torch.cat([y, z], dim=-1)
        y_update = self.network(y_input)
        y = y + y_update
        
        return y, z


class TransformerNetwork(nn.Module):
    """Transformer-based network for larger context lengths"""
    def __init__(self, context_length, hidden_size, n_layers=2):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Pre-create projections for known input sizes
        self.proj_3d = nn.Linear(hidden_size * 3, hidden_size)
        self.proj_2d = nn.Linear(hidden_size * 2, hidden_size)
        
        nn.init.normal_(self.proj_3d.weight, std=0.02)
        nn.init.zeros_(self.proj_3d.bias)
        nn.init.normal_(self.proj_2d.weight, std=0.02)
        nn.init.zeros_(self.proj_2d.bias)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, n_heads=4)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        
        # Project based on input size
        if input_dim == self.hidden_size * 3:
            x = self.proj_3d(x)
        elif input_dim == self.hidden_size * 2:
            x = self.proj_2d(x)
        else:
            raise ValueError(f"Unexpected input size: {input_dim}")
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        return x


class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention"""
    def __init__(self, hidden_size, n_heads=4):
        super().__init__()
        
        assert hidden_size % n_heads == 0
        
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        
        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        
        # FFN
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Initialize
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)
        for module in self.ffn.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        
        B, L, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, n_heads, L, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, L, D)
        x = self.proj(x)
        x = residual + x
        
        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


# Keep the TinyNetwork class for MLP-Mixer option
class TinyNetwork(nn.Module):
    """MLP-Mixer style network (for smaller context)"""
    def __init__(self, context_length, hidden_size, n_layers=2):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Pre-create projections
        self.proj_3d = nn.Linear(hidden_size * 3, hidden_size)
        self.proj_2d = nn.Linear(hidden_size * 2, hidden_size)
        
        nn.init.normal_(self.proj_3d.weight, std=0.02)
        nn.init.zeros_(self.proj_3d.bias)
        nn.init.normal_(self.proj_2d.weight, std=0.02)
        nn.init.zeros_(self.proj_2d.bias)
        
        # Mixer layers
        self.layers = nn.ModuleList([
            MixerLayer(context_length, hidden_size)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        
        # Project based on input size
        if input_dim == self.hidden_size * 3:
            x = self.proj_3d(x)
        elif input_dim == self.hidden_size * 2:
            x = self.proj_2d(x)
        else:
            raise ValueError(f"Unexpected input size: {input_dim}")
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        return x


class MixerLayer(nn.Module):
    def __init__(self, context_length, hidden_size):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Token mixing
        self.token_mixing = nn.Sequential(
            nn.Linear(context_length, context_length * 2),
            nn.GELU(),
            nn.Linear(context_length * 2, context_length)
        )
        
        # Channel mixing
        self.channel_mixing = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Initialize
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, x):
        # Token mixing
        residual = x
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = self.token_mixing(x)
        x = x.transpose(1, 2)
        x = residual + x
        
        # Channel mixing
        residual = x
        x = self.norm2(x)
        x = self.channel_mixing(x)
        x = residual + x
        
        return x


# Test the model
if __name__ == "__main__":
    print("Testing TRM model...")
    
    # Create model
    model = TinyRecursiveModel(
        vocab_size=10,
        context_length=81,
        hidden_size=256,  # Increased from 128
        n_layers=2,
        n_recursions=6,
        T_cycles=3,
        use_attention=True  # Use Transformer
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params:,} parameters")
    
    # Test forward pass
    batch_size = 4
    x = torch.randint(0, 5, (batch_size, 16))  # Random puzzles
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    predictions, q_values, final_y, final_z = model(x, n_supervision_steps=3, training=True)
    
    print(f"Number of supervision steps: {len(predictions)}")
    print(f"Prediction shape per step: {predictions[0].shape}")
    print(f"Q-value shape per step: {q_values[0].shape}")
    print(f"Final y shape: {final_y.shape}")
    print(f"Final z shape: {final_z.shape}")
    
    # Test inference mode
    print("\nTesting inference mode...")
    with torch.no_grad():
        predictions, q_values, final_y, final_z = model(x, n_supervision_steps=16, training=False)
    print(f"Stopped after {len(predictions)} steps (max 16)")
    
    print("\nModel test complete!")
