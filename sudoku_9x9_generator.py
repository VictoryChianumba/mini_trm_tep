# sudoku_9x9_generator.py

import numpy as np
import random

def generate_9x9_sudoku(max_attempts=100):
    """Generate a valid 9×9 Sudoku puzzle and solution"""
    
    def is_valid(grid, row, col, num):
        # Check row
        if num in grid[row]:
            return False
        # Check column
        if num in grid[:, col]:
            return False
        # Check 3×3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        if num in grid[box_row:box_row+3, box_col:box_col+3]:
            return False
        return True
    
    def solve(grid):
        for i in range(9):
            for j in range(9):
                if grid[i, j] == 0:
                    nums = list(range(1, 10))
                    random.shuffle(nums)
                    for num in nums:
                        if is_valid(grid, i, j, num):
                            grid[i, j] = num
                            if solve(grid):
                                return True
                            grid[i, j] = 0
                    return False
        return True
    
    # Try to generate a valid solution
    for attempt in range(max_attempts):
        solution = np.zeros((9, 9), dtype=np.int32)
        if solve(solution):
            # Create puzzle by removing numbers
            puzzle = solution.copy()
            positions = [(i, j) for i in range(9) for j in range(9)]
            random.shuffle(positions)
            
            # Remove 40-50 numbers (leaving 31-41 clues)
            # This creates medium-hard puzzles
            num_to_remove = random.randint(40, 50)
            for i in range(num_to_remove):
                row, col = positions[i]
                puzzle[row, col] = 0
            
            return puzzle, solution
    
    raise Exception(f"Failed to generate valid sudoku after {max_attempts} attempts")


def augment_9x9_sudoku(puzzle, solution, n_augmentations=100):
    """Generate augmented versions of a 9×9 sudoku puzzle"""
    augmented_puzzles = []
    augmented_solutions = []
    
    for _ in range(n_augmentations):
        # Number permutation (shuffle which digit is which)
        perm = np.random.permutation([1, 2, 3, 4, 5, 6, 7, 8, 9])
        mapping = {i+1: perm[i] for i in range(9)}
        mapping[0] = 0  # Keep empty cells as 0
        
        aug_puzzle = np.vectorize(mapping.get)(puzzle).copy()
        aug_solution = np.vectorize(mapping.get)(solution).copy()
        
        # Row swaps within bands (rows 0-2, 3-5, 6-8)
        if random.random() < 0.5:
            band = random.choice([0, 3, 6])
            rows = [band, band+1, band+2]
            random.shuffle(rows)
            aug_puzzle[band:band+3] = aug_puzzle[rows]
            aug_solution[band:band+3] = aug_solution[rows]
        
        # Column swaps within stacks (cols 0-2, 3-5, 6-8)
        if random.random() < 0.5:
            stack = random.choice([0, 3, 6])
            cols = [stack, stack+1, stack+2]
            random.shuffle(cols)
            aug_puzzle[:, stack:stack+3] = aug_puzzle[:, cols]
            aug_solution[:, stack:stack+3] = aug_solution[:, cols]
        
        # Transpose
        if random.random() < 0.5:
            aug_puzzle = aug_puzzle.T.copy()
            aug_solution = aug_solution.T.copy()
        
        augmented_puzzles.append(aug_puzzle.flatten())
        augmented_solutions.append(aug_solution.flatten())
    
    return augmented_puzzles, augmented_solutions


def generate_dataset(n_base_puzzles=1000, n_augmentations=100, seed=42):
    """Generate dataset with augmentations"""
    random.seed(seed)
    np.random.seed(seed)
    
    all_puzzles = []
    all_solutions = []
    
    print(f"Generating {n_base_puzzles} base 9×9 puzzles...")
    for i in range(n_base_puzzles):
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{n_base_puzzles}")
        
        puzzle, solution = generate_9x9_sudoku()
        
        # Add augmentations
        aug_puzzles, aug_solutions = augment_9x9_sudoku(puzzle, solution, n_augmentations)
        all_puzzles.extend(aug_puzzles)
        all_solutions.extend(aug_solutions)
    
    print(f"Total dataset size: {len(all_puzzles)} examples")
    return np.array(all_puzzles), np.array(all_solutions)