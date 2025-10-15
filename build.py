import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from copy import deepcopy

class SudokuDataset(Dataset):
    """Dataset for Sudoku puzzles"""
    def __init__(self, puzzles, solutions):
        self.puzzles = torch.from_numpy(puzzles).long()
        self.solutions = torch.from_numpy(solutions).long()
    
    def __len__(self):
        return len(self.puzzles)
    
    def __getitem__(self, idx):
        return self.puzzles[idx], self.solutions[idx]


class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.cpu().clone()
    
    def update(self):
        """Update shadow parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Add new parameters if they don't exist yet
                if name not in self.shadow:
                    self.shadow[name] = param.data.cpu().clone()
                else:
                    self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data.cpu()
    
    def apply_shadow(self):
        """Apply shadow parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                if name in self.shadow:
                    param.data = self.shadow[name].to(param.device) 
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}

def compute_loss(predictions, q_values, target, n_supervision_steps):
    """
    Compute loss for deep supervision
    
    predictions: list of [B, L, vocab_size] tensors (one per supervision step)
    q_values: list of [B, 1] tensors (one per supervision step)
    target: [B, L] ground truth
    """
    total_loss = 0.0
    
    for step_idx, (pred, q) in enumerate(zip(predictions, q_values)):
        # Prediction loss (cross-entropy)
        pred_loss = F.cross_entropy(
            pred.reshape(-1, pred.size(-1)),  # [B*L, vocab_size]
            target.reshape(-1)                 # [B*L]
        )
        
        # Check if prediction is correct
        pred_tokens = pred.argmax(dim=-1)  # [B, L]
        is_correct = (pred_tokens == target).all(dim=1).float()  # [B]
        
        # Halting loss (binary cross-entropy)
        # Q-value should predict if current answer is correct
        halt_loss = F.binary_cross_entropy_with_logits(
            q.squeeze(-1),  # [B]
            is_correct      # [B]
        )
        
        # Combine losses
        step_loss = pred_loss + halt_loss
        total_loss += step_loss.item()
    
    # Average over supervision steps
    return total_loss / len(predictions)


def train_epoch(model, dataloader, optimizer, ema, device, n_supervision_steps=16):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_steps_taken = 0
    
    for batch_idx, (puzzles, solutions) in enumerate(dataloader):
        puzzles = puzzles.to(device)
        solutions = solutions.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions, q_values, _, _ = model(
            puzzles, 
            n_supervision_steps=n_supervision_steps,
            training=True
        )
        
        # Compute loss
        loss = compute_loss(predictions, q_values, solutions, n_supervision_steps)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update EMA
        ema.update()
        
        # Track metrics
        total_loss += loss.item()
        
        # Check final prediction accuracy
        final_pred = predictions[-1].argmax(dim=-1)  # [B, L]
        correct = (final_pred == solutions).all(dim=1).sum().item()
        total_correct += correct
        total_samples += puzzles.size(0)
        total_steps_taken += len(predictions)
        
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            avg_acc = total_correct / total_samples
            avg_steps = total_steps_taken / total_samples
            print(f"  Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Acc: {avg_acc:.4f} | "
                  f"Avg Steps: {avg_steps:.2f}")
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_samples
    avg_steps = total_steps_taken / total_samples
    
    return avg_loss, avg_acc, avg_steps


def evaluate(model, dataloader, device, n_supervision_steps=16):
    """Evaluate model"""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_steps_taken = 0
    
    with torch.no_grad():
        for puzzles, solutions in dataloader:
            puzzles = puzzles.to(device)
            solutions = solutions.to(device)
            
            # Forward pass
            predictions, q_values, _, _ = model(
                puzzles,
                n_supervision_steps=n_supervision_steps,
                training=False  # Use early stopping
            )
            
            # Check accuracy
            final_pred = predictions[-1].argmax(dim=-1)
            correct = (final_pred == solutions).all(dim=1).sum().item()
            total_correct += correct
            total_samples += puzzles.size(0)
            total_steps_taken += len(predictions)
    
    accuracy = total_correct / total_samples
    avg_steps = total_steps_taken / total_samples
    
    return accuracy, avg_steps


def train(
    model,
    train_dataset,
    val_dataset,
    n_epochs=100,
    batch_size=768,
    learning_rate=1e-4,
    weight_decay=1.0,
    n_supervision_steps=16,
    device='cpu',
    save_path='trm_sudoku.pt'
):
    """Full training loop"""
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for M1 compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay
    )
    
    # Learning rate schedule (warmup + cosine decay)
    warmup_steps = 2000
    total_steps = n_epochs * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # EMA
    ema = EMA(model, decay=0.999)
    
    # Training loop
    best_val_acc = 0.0
    step = 0
    
    print(f"Starting training...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {n_epochs}")
    print(f"Device: {device}\n")
    
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        
        # Train
        train_loss, train_acc, train_steps = train_epoch(
            model, train_loader, optimizer, ema, device, n_supervision_steps
        )
        
        print(f"  Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Avg Steps: {train_steps:.2f}")
        
        # Update learning rate
        for _ in range(len(train_loader)):
            scheduler.step()
            step += 1
        
        # Evaluate with EMA weights
        if (epoch + 1) % 5 == 0:  # Evaluate every 5 epochs
            ema.apply_shadow()
            val_acc, val_steps = evaluate(model, val_loader, device, n_supervision_steps)
            ema.restore()
            
            print(f"  Val Acc: {val_acc:.4f} | Avg Steps: {val_steps:.2f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ema.apply_shadow()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_path)
                ema.restore()
                print(f"  Saved best model (val_acc: {val_acc:.4f})")
        
        print()
    
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    return model


# Main training script
if __name__ == "__main__":
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Generate data (or load if you already generated it)
    print("Generating dataset...")
    from data_generation import generate_dataset  # Assuming you saved the data gen code
    from trm import TinyRecursiveModel
    
    train_puzzles, train_solutions = generate_dataset(
        n_base_puzzles=200, 
        n_augmentations=50,
        seed=42
    )
    val_puzzles, val_solutions = generate_dataset(
        n_base_puzzles=100,
        n_augmentations=100,
        seed=43
    )
    
    # Create datasets
    train_dataset = SudokuDataset(train_puzzles, train_solutions)
    val_dataset = SudokuDataset(val_puzzles, val_solutions)
    
    # Create model
    model = TinyRecursiveModel(
        vocab_size=5,
        context_length=16,
        hidden_size=128,
        n_layers=2,
        n_recursions=6,
        T_cycles=3
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
   
    from train_with_logging import train_with_logging  
    model, logger = train_with_logging(
        model,
        train_dataset,
        val_dataset,
        n_epochs=60,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=1.0,
        n_supervision_steps=8,
        device=device,
        save_path='trm_4x4_sudoku.pt',
        log_dir='logs',
        experiment_name='trm_4x4_baseline'
    )
