import torch
import numpy as np
from trm import TinyRecursiveModel
from sudoku_6x6_generator import generate_dataset
from build import SudokuDataset
from train_step_by_step import train_step_by_step

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load data
print("Loading data...")
# Generate data
print("Generating dataset...")
train_puzzles, train_solutions = generate_dataset(
    n_base_puzzles=1000,
    n_augmentations=500,
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


# Create model - NOW WE CAN USE LARGER SIZES!
model = TinyRecursiveModel(
    vocab_size=7,
    context_length=36,
    hidden_size=512,            # Can use paper's size now!
    n_layers=2,
    n_recursions=6,
    T_cycles=3,
    use_attention=False
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# Train with step-by-step backward
model = train_step_by_step(
    model,
    train_dataset,
    val_dataset,
    n_epochs=2000,
    batch_size=512,              # Can use larger batch now!
    learning_rate=1e-4,
    weight_decay=1.0,
    n_supervision_steps=16,      # Can use all 16 steps now!
    device=device,
    save_path='trm_6x6_memory_efficient.pt'
)