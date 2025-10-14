# train_with_logging.py - Modified training loop with comprehensive logging

import torch
import time
from logger import TrainingLogger
from matplotlib import pyplot as plt
from evaluation import (evaluate_with_step_breakdown, 
                       collect_example_predictions,
                       evaluate_error_analysis)
from visualization import (visualize_step_by_step, 
                          create_solution_animation,
                          visualize_attention_heatmap)
from demo import generate_html_demo

def train_with_logging(
    model,
    train_dataset,
    val_dataset,
    n_epochs=60,
    batch_size=32,
    learning_rate=1e-4,
    weight_decay=1.0,
    n_supervision_steps=8,
    device='cpu',
    save_path='trm_sudoku.pt',
    log_dir='logs',
    experiment_name=None
):
    """Training loop with comprehensive logging"""
    
    # Initialize logger
    logger = TrainingLogger(log_dir=log_dir, experiment_name=experiment_name)
    
    # Log configuration
    config = {
        'model_params': sum(p.numel() for p in model.parameters()),
        'hidden_size': model.hidden_size,
        'n_layers': model.n_layers,
        'n_recursions': model.n_recursions,
        'T_cycles': model.T_cycles,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'n_supervision_steps': n_supervision_steps,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
    }
    logger.log_config(config)
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay
    )
    
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
    
    # EMA
    from build import EMA
    ema = EMA(model, decay=0.999)
    
    best_val_acc = 0.0
    
    print("Starting training with logging...")
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        
        # Train
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        train_steps_sum = 0
        
        for puzzles, solutions in train_loader:
            puzzles = puzzles.to(device)
            solutions = solutions.to(device)
            
            optimizer.zero_grad()
            
            predictions, q_values, _, _ = model(
                puzzles, 
                n_supervision_steps=n_supervision_steps,
                training=True
            )
            
            # Compute loss
            from build import compute_loss
            loss = compute_loss(predictions, q_values, solutions, n_supervision_steps)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update()
            scheduler.step()
            
            # Track metrics
            train_loss_sum += loss.item()
            final_pred = predictions[-1].argmax(dim=-1)
            train_correct += (final_pred == solutions).all(dim=1).sum().item()
            train_total += puzzles.size(0)
            train_steps_sum += len(predictions)
        
        train_loss = train_loss_sum / len(train_loader)
        train_acc = train_correct / train_total
        train_avg_steps = train_steps_sum / train_total
        
        # Evaluate with step breakdown
        ema.apply_shadow()
        val_acc, val_avg_steps, val_step_accs, val_per_pos_acc = evaluate_with_step_breakdown(
        model, val_loader, device, n_supervision_steps
        )
        
        # Compute validation loss (optional, takes time)
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for puzzles, solutions in val_loader:
                puzzles = puzzles.to(device)
                solutions = solutions.to(device)
                predictions, q_values, _, _ = model(puzzles, n_supervision_steps=n_supervision_steps, training=False)
                from build import compute_loss
                loss = compute_loss(predictions, q_values, solutions, n_supervision_steps)
                val_loss_sum += loss.item()
        val_loss = val_loss_sum / len(val_loader)
        
        ema.restore()
        
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]
        
        # Log epoch
        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            lr=current_lr,
            avg_steps_train=train_avg_steps,
            avg_steps_val=val_avg_steps,
            epoch_time=epoch_time,
            val_step_accs=val_step_accs
        )
        
        print(f"Epoch {epoch+1}/{n_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val Per-Pos: {val_per_pos_acc:.4f} | "  # ADD THIS
            f"Val Steps: {val_avg_steps:.2f} | Time: {epoch_time:.1f}s"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ema.apply_shadow()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config,
            }, save_path)
            ema.restore()
            print(f"  Saved best model (val_acc: {val_acc:.4f})")
        
        # # Collect examples every 5 epochs
        # if (epoch + 1) % 5 == 0:
        #     ema.apply_shadow()
        #     examples = collect_example_predictions(
        #         model, val_loader, device, n_examples=10, n_supervision_steps=n_supervision_steps
        #     )
        #     logger.log_examples(epoch, examples)
            
            # Create visualizations for first few examples
            # for i, ex in enumerate(examples[:3]):
            #     fig = visualize_step_by_step(
            #         ex, 
            #         save_path=logger.log_dir / f'example_{i}_epoch_{epoch}.png'
            #     )
            #     plt.close(fig)
            
            # ema.restore()
    
    # Final evaluation and visualization
    print("\nTraining complete! Generating final visualizations...")
    
    # Plot training curves
    logger.plot_training_curves(save=True)
    logger.plot_step_accuracies(save=True)
    
    # Generate summary
    summary = logger.generate_summary()
    
    # Collect final examples
    ema.apply_shadow()
    # final_examples = collect_example_predictions(
    #     model, val_loader, device, n_examples=10, n_supervision_steps=16
    # )
    
    # Create animations for a few examples
    # for i, ex in enumerate(final_examples[:3]):
    #     create_solution_animation(
    #         ex,
    #         save_path=logger.log_dir / f'animation_{i}.gif',
    #         fps=2
    #     )
    
    # Generate interactive HTML demo
    # generate_html_demo(
    #     save_path,
    #     final_examples,
    #     output_path=logger.log_dir / 'demo.html'
    # )
    
    # Error analysis
    errors = evaluate_error_analysis(model, val_loader, device, n_supervision_steps=16)
    print(f"\nError Analysis: {len(errors)} errors out of {len(val_dataset)} ({100*len(errors)/len(val_dataset):.1f}%)")
    
    ema.restore()
    
    print(f"\n✓ All results saved to: {logger.log_dir}")
    print(f"✓ Open {logger.log_dir / 'demo.html'} in a browser to try the interactive demo!")
    
    return model, logger