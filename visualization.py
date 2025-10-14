# visualization.py - Visualize solutions step-by-step

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
import torch

def visualize_sudoku_grid(grid, grid_size=4, ax=None, title=None):
    """Draw a single Sudoku grid"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    box_size = int(np.sqrt(grid_size))
    grid = grid.reshape(grid_size, grid_size)
    
    # Draw grid
    for i in range(grid_size + 1):
        linewidth = 2 if i % box_size == 0 else 0.5
        ax.plot([i, i], [0, grid_size], 'k-', linewidth=linewidth)
        ax.plot([0, grid_size], [i, i], 'k-', linewidth=linewidth)
    
    # Fill in numbers
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i, j] != 0:
                ax.text(j + 0.5, grid_size - i - 0.5, str(grid[i, j]),
                       ha='center', va='center', fontsize=20, fontweight='bold')
    
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    return ax


def visualize_step_by_step(example, save_path=None):
    """
    Visualize how the model's prediction evolves through supervision steps
    """
    puzzle = example['puzzle']
    predictions = example['predictions']
    target = example['target']
    q_values = example['q_values']
    
    n_steps = len(predictions)
    grid_size = int(np.sqrt(len(puzzle)))
    
    # Create figure with subplots
    n_cols = min(5, n_steps + 2)  # Max 5 columns
    n_rows = (n_steps + 2 + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten() if n_steps + 2 > 1 else [axes]
    
    # Input puzzle
    visualize_sudoku_grid(puzzle, grid_size, axes[0], 'Input Puzzle')
    
    # Predictions at each step
    for step, (pred, q) in enumerate(zip(predictions, q_values)):
        if step + 1 < len(axes) - 1:
            is_correct = np.array_equal(pred, target)
            color = 'green' if is_correct else 'red'
            title = f'Step {step}\nQ={q:.3f}'
            
            visualize_sudoku_grid(pred, grid_size, axes[step + 1], title)
            axes[step + 1].set_title(title, color=color, fontsize=14, fontweight='bold')
    
    # Target
    if len(axes) > n_steps + 1:
        visualize_sudoku_grid(target, grid_size, axes[n_steps + 1], 'Target')
    
    # Hide unused subplots
    for i in range(n_steps + 2, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig


def create_solution_animation(example, save_path=None, fps=2):
    """
    Create an animated GIF showing the solution process
    """
    puzzle = example['puzzle']
    predictions = example['predictions']
    target = example['target']
    grid_size = int(np.sqrt(len(puzzle)))
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    def update(frame):
        ax.clear()
        
        if frame == 0:
            # Show puzzle
            visualize_sudoku_grid(puzzle, grid_size, ax, f'Input Puzzle')
        elif frame <= len(predictions):
            # Show prediction at this step
            pred = predictions[frame - 1]
            is_correct = np.array_equal(pred, target)
            title = f'Step {frame-1}' + (' ✓' if is_correct else '')
            visualize_sudoku_grid(pred, grid_size, ax, title)
        else:
            # Show target
            visualize_sudoku_grid(target, grid_size, ax, 'Target Solution')
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(predictions) + 2,
        interval=1000//fps, repeat=True
    )
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Saved animation to {save_path}")
    
    plt.close()
    return anim


def visualize_attention_heatmap(model, puzzle, device, grid_size=4):
    """
    Visualize what the model is "attending to" at each position
    This is a simplified version - true attention would need hooks into the model
    """
    # This is a placeholder - would need to extract actual attention weights
    # or activations from the model
    
    model.eval()
    with torch.no_grad():
        puzzle_tensor = torch.from_numpy(puzzle).unsqueeze(0).to(device)
        
        # Get model's latent states
        predictions, q_values, y_state, z_state = model(
            puzzle_tensor,
            n_supervision_steps=8,
            training=False
        )
        
        # Use z_state magnitude as a proxy for "attention"
        z_final = z_state[0].cpu().numpy()  # [L, D]
        attention_proxy = np.linalg.norm(z_final, axis=1)  # [L]
        attention_map = attention_proxy.reshape(grid_size, grid_size)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show puzzle
    visualize_sudoku_grid(puzzle, grid_size, ax1, 'Input Puzzle')
    
    # Show attention heatmap
    im = ax2.imshow(attention_map, cmap='hot', interpolation='nearest')
    ax2.set_title('Model "Attention"\n(Z-state magnitude)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Overlay grid
    box_size = int(np.sqrt(grid_size))
    for i in range(grid_size + 1):
        linewidth = 2 if i % box_size == 0 else 0.5
        ax2.plot([i-0.5, i-0.5], [-0.5, grid_size-0.5], 'w-', linewidth=linewidth, alpha=0.5)
        ax2.plot([-0.5, grid_size-0.5], [i-0.5, i-0.5], 'w-', linewidth=linewidth, alpha=0.5)
    
    plt.tight_layout()
    return fig