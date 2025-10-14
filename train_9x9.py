# train_9x9.py

import torch
from sudoku_9x9_generator import generate_dataset
from trm import TinyRecursiveModel
from train_with_logging import train_with_logging
from build import SudokuDataset

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Generate data - START SMALL for testing
print("Generating dataset...")
train_puzzles, train_solutions = generate_dataset(
    n_base_puzzles=100,  # Start with 100, scale to 1000 if it works
    n_augmentations=100,
    seed=42
)
val_puzzles, val_solutions = generate_dataset(
    n_base_puzzles=20,
    n_augmentations=50,
    seed=43
)

# Create datasets
train_dataset = SudokuDataset(train_puzzles, train_solutions)
val_dataset = SudokuDataset(val_puzzles, val_solutions)

# Create model
model = TinyRecursiveModel(
    vocab_size=10,
    context_length=81,
    hidden_size=256,  # Increased from 128
    n_layers=2,
    n_recursions=6,
    T_cycles=3,
    use_attention=True  # Use Transformer
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}\n")

# Train
model, logger = train_with_logging(
    model,
    train_dataset,
    val_dataset,
    n_epochs=100,  # More epochs for harder task
    batch_size=16,  # Smaller batch size (larger context)
    learning_rate=1e-4,
    weight_decay=0.1,  # Reduced from 1.0
    n_supervision_steps=16,  # More steps for harder task
    device=device,
    save_path='trm_9x9_sudoku.pt',
    log_dir='logs',
    experiment_name='trm_9x9_baseline'
)