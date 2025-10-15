import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time    
from build import EMA

def train_step_by_step(
    model,
    train_dataset,
    val_dataset,
    n_epochs=100,
    batch_size=256,
    learning_rate=1e-4,
    weight_decay=1.0,
    n_supervision_steps=16,
    device='cpu',
    save_path='model.pt'
):
    """
    Training loop with step-by-step backward to save memory
    """
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay
    )
    
    # Learning rate schedule
    warmup_steps = 2000
    total_steps = n_epochs * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            import numpy as np
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # EMA with CPU shadows (saves GPU memory)
    ema = EMA(model, decay=0.999)
    
    best_val_acc = 0.0
    global_step = 0
    
    print(f"Training with step-by-step backward...")
    print(f"This saves ~{n_supervision_steps}× memory!\n")
    
    for epoch in range(n_epochs):
        model.train()
        epoch_start = time.time()
        
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (puzzles, solutions) in enumerate(train_loader):
            puzzles = puzzles.to(device)
            solutions = solutions.to(device)
            
            # Embed input once
            x_embedded = model.input_embedding(puzzles)
            
            # Initialize y and z
            batch_size_actual = puzzles.size(0)
            y = model.y_init.expand(batch_size_actual, -1, -1).clone()
            z = model.z_init.expand(batch_size_actual, -1, -1).clone()
            
            # Zero gradients once at start
            optimizer.zero_grad()
            
            # Step-by-step forward and backward
            total_loss = 0.0
            final_pred = None
            
            for step_idx in range(n_supervision_steps):
                # Forward ONE step (keeps gradients)
                y, z, y_logits, q = model.forward_single_step(x_embedded, y, z)
                
                # Compute loss for this step
                pred_loss = F.cross_entropy(
                    y_logits.reshape(-1, y_logits.size(-1)),
                    solutions.reshape(-1)
                )
                
                # Halting loss()
                pred_tokens = y_logits.argmax(dim=-1)
                is_correct = (pred_tokens == solutions).all(dim=1).float()
                halt_loss = F.binary_cross_entropy_with_logits(q.squeeze(-1), is_correct)
                
                # Step loss (normalized by number of steps)
                step_loss = (pred_loss + halt_loss) / n_supervision_steps
                
                # Backward immediately (frees graph after this)
                retrain =  (step_idx < n_supervision_steps -1)
                step_loss.backward(retain_graph = retrain)
                
                # Accumulate loss for logging
                total_loss += step_loss.item()
                
                # Save final prediction
                if step_idx == n_supervision_steps - 1:
                    final_pred = pred_tokens
                
                # CRITICAL: Detach y and z for next step
                # This breaks the computation graph between steps
                y = y.detach()
                z = z.detach()
            
            # Clip gradients (accumulated from all steps)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters (once for all steps)
            optimizer.step()
            scheduler.step()
            
            # Update EMA
            ema.update()
            
            # Track metrics
            train_loss_sum += total_loss
            train_correct += (final_pred == solutions).all(dim=1).sum().item()
            train_total += puzzles.size(0)
            
            global_step += 1
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                avg_loss = train_loss_sum / (batch_idx + 1)
                avg_acc = train_correct / train_total
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
        
        # Epoch metrics
        train_loss = train_loss_sum / len(train_loader)
        train_acc = train_correct / train_total
        epoch_time = time.time() - epoch_start
        
        # Validation
        if (epoch + 1) % 5 == 0:
            val_acc, val_per_pos = evaluate_step_by_step(model, val_loader, device, n_supervision_steps)
            
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val Per-Pos: {val_per_pos:.4f} | "
                  f"Time: {epoch_time:.1f}s")
            
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
        else:
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Time: {epoch_time:.1f}s")
    
    print(f"\nTraining complete! Best val accuracy: {best_val_acc:.4f}")
    return model


def evaluate_step_by_step(model, dataloader, device, n_supervision_steps=16):
    """Evaluation with step-by-step forward (no backward, so memory is fine)"""
    model.eval()
    
    total_correct = 0
    total_samples = 0
    per_pos_correct = 0
    per_pos_total = 0
    
    with torch.no_grad():
        for puzzles, solutions in dataloader:
            puzzles = puzzles.to(device)
            solutions = solutions.to(device)
            
            # Can use regular forward for evaluation (no backward needed)
            predictions, q_values, _, _ = model(
                puzzles,
                n_supervision_steps=n_supervision_steps,
                training=False
            )
            
            # Final prediction
            final_pred = predictions[-1].argmax(dim=-1)
            
            # Full puzzle accuracy
            correct = (final_pred == solutions).all(dim=1).sum().item()
            total_correct += correct
            total_samples += puzzles.size(0)
            
            # Per-position accuracy
            per_pos_correct += (final_pred == solutions).sum().item()
            per_pos_total += final_pred.numel()
    
    accuracy = total_correct / total_samples
    per_pos_acc = per_pos_correct / per_pos_total
    
    return accuracy, per_pos_acc