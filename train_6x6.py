# train_6x6.py

import torch
from sudoku_6x6_generator import generate_dataset
from trm import TinyRecursiveModel  # Reuse the model, just change params
from train_with_logging import train_with_logging
from build import SudokuDataset

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cuda')

print(f"Using device: {device}\n")


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

# Create model
model = TinyRecursiveModel(
    vocab_size=7,        # 0-6 for 6×6 Sudoku
    context_length=36,   # 6×6 grid flattened
    hidden_size=384,
    n_layers=2,
    n_recursions=6,
    T_cycles=3,
    use_attention=False   # Use Transformer (should be manageable for L=36)
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}\n")

# Train
model, logger = train_with_logging(
    model,
    train_dataset,
    val_dataset,
    n_epochs=1000,
    batch_size =256, # close to the paper value of 768
    learning_rate=1e-4,
    weight_decay=1.0,
    n_supervision_steps=16,  # Slightly more than 4×4, less than 9×9
    device=device,
    save_path='trm_6x6_sudoku.pt',
    log_dir='logs',
    experiment_name='trm_6x6_baseline'
)
