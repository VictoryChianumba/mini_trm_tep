# mini_trm_tep

## Data Generation

Generate the training data with:

```bash
python -c "
from sudoku_6x6_generator import generate_dataset
import numpy as np

# Generate training data
train_puzzles, train_solutions = generate_dataset(
    n_base_puzzles=1000,
    n_augmentations=500,
    seed=42
)
np.save('train_puzzles_6x6_500k.npy', train_puzzles)
np.save('train_solutions_6x6_500k.npy', train_solutions)

# Generate validation data
val_puzzles, val_solutions = generate_dataset(
    n_base_puzzles=100,
    n_augmentations=100,
    seed=43
)
np.save('val_puzzles_6x6_10k.npy', val_puzzles)
np.save('val_solutions_6x6_10k.npy', val_solutions)
"
```

This takes ~30 minutes to generate.
