# logger.py - Comprehensive logging system

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from datetime import datetime

class TrainingLogger:
    """Logs training metrics, examples, and creates visualizations"""
    
    def __init__(self, log_dir='logs', experiment_name=None):
        if experiment_name is None:
            experiment_name = f"trm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training metrics
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'avg_steps_train': [],
            'avg_steps_val': [],
            'time_per_epoch': [],
        }
        
        # Per-step accuracy breakdown
        self.step_accuracies = {
            'train': [],  # List of lists: [[step0_acc, step1_acc, ...], ...]
            'val': []
        }
        
        # Example predictions
        self.examples = {}
        
        # Model config
        self.config = {}
        
        print(f"Logging to: {self.log_dir}")
    
    def log_config(self, config):
        """Save model and training configuration"""
        self.config = config
        with open(self.log_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, 
                  lr, avg_steps_train, avg_steps_val, epoch_time,
                  train_step_accs=None, val_step_accs=None):
        """Log metrics for one epoch"""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['learning_rate'].append(lr)
        self.history['avg_steps_train'].append(avg_steps_train)
        self.history['avg_steps_val'].append(avg_steps_val)
        self.history['time_per_epoch'].append(epoch_time)
        
        if train_step_accs is not None:
            self.step_accuracies['train'].append(train_step_accs)
        if val_step_accs is not None:
            self.step_accuracies['val'].append(val_step_accs)
        
        # Save history after each epoch
        self.save_history()
    
    def log_examples(self, epoch, examples):
        """
        Save example predictions
        examples: list of (puzzle, predictions_per_step, target) tuples
        """
        self.examples[f'epoch_{epoch}'] = examples
        
        # Save examples
        np.save(self.log_dir / f'examples_epoch_{epoch}.npy', examples, allow_pickle=True)
    
    def save_history(self):
        """Save training history to JSON"""
        with open(self.log_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Also save step accuracies
        with open(self.log_dir / 'step_accuracies.json', 'w') as f:
            json.dump(self.step_accuracies, f, indent=2)
    
    def plot_training_curves(self, save=True):
        """Generate training curve plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        epochs = self.history['epoch']
        
        # Loss curves
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.history['train_acc'], label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_acc'], label='Val', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[0, 2].plot(epochs, self.history['learning_rate'], linewidth=2, color='green')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_title('Learning Rate Schedule')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_yscale('log')
        
        # Avg steps
        axes[1, 0].plot(epochs, self.history['avg_steps_train'], label='Train', linewidth=2)
        axes[1, 0].plot(epochs, self.history['avg_steps_val'], label='Val', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Avg Supervision Steps')
        axes[1, 0].set_title('Average Supervision Steps Used')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Time per epoch
        axes[1, 1].plot(epochs, self.history['time_per_epoch'], linewidth=2, color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_title('Time per Epoch')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Train/Val gap
        gap = np.array(self.history['train_acc']) - np.array(self.history['val_acc'])
        axes[1, 2].plot(epochs, gap, linewidth=2, color='red')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Accuracy Gap')
        axes[1, 2].set_title('Generalization Gap (Train - Val)')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.log_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
            print(f"Saved training curves to {self.log_dir / 'training_curves.png'}")
        
        return fig
    
    def plot_step_accuracies(self, save=True):
        """Plot accuracy at each supervision step over training"""
        if not self.step_accuracies['val']:
            print("No step accuracy data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Validation step accuracies over time
        val_step_accs = np.array(self.step_accuracies['val'])  # [n_epochs, n_steps]
        n_epochs, n_steps = val_step_accs.shape
        
        for step in range(n_steps):
            ax1.plot(range(n_epochs), val_step_accs[:, step], 
                    label=f'Step {step}', linewidth=2, marker='o', markersize=3)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Validation Accuracy per Supervision Step')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Final epoch: accuracy by step
        if len(self.step_accuracies['val']) > 0:
            final_val_accs = self.step_accuracies['val'][-1]
            ax2.bar(range(len(final_val_accs)), final_val_accs, color='steelblue')
            ax2.set_xlabel('Supervision Step')
            ax2.set_ylabel('Accuracy')
            ax2.set_title(f'Final Validation Accuracy by Step (Epoch {n_epochs-1})')
            ax2.set_xticks(range(len(final_val_accs)))
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.log_dir / 'step_accuracies.png', dpi=300, bbox_inches='tight')
            print(f"Saved step accuracies to {self.log_dir / 'step_accuracies.png'}")
        
        return fig
    
    def generate_summary(self):
        """Generate a summary report"""
        if not self.history['epoch']:
            print("No training data to summarize")
            return
        
        summary = {
            'experiment_name': self.log_dir.name,
            'total_epochs': len(self.history['epoch']),
            'best_val_acc': max(self.history['val_acc']),
            'best_val_epoch': self.history['epoch'][np.argmax(self.history['val_acc'])],
            'final_train_acc': self.history['train_acc'][-1],
            'final_val_acc': self.history['val_acc'][-1],
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'total_training_time': sum(self.history['time_per_epoch']),
            'avg_time_per_epoch': np.mean(self.history['time_per_epoch']),
            'generalization_gap': self.history['train_acc'][-1] - self.history['val_acc'][-1],
        }
        
        # Save summary
        with open(self.log_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key:.<40} {value:.4f}")
            else:
                print(f"{key:.<40} {value}")
        print("="*60 + "\n")
        
        return summary