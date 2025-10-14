# sudoku_6x6_generator.py

import numpy as np
import random

def generate_6x6_sudoku(max_attempts=100):
    """Generate a valid 6×6 Sudoku puzzle and solution"""
    
    def is_valid(grid, row, col, num):
        # Check row
        if num in grid[row]:
            return False
        # Check column
        if num in grid[:, col]:
            return False
        # Check 2×3 box
        box_row, box_col = 2 * (row // 2), 3 * (col // 3)
        if num in grid[box_row:box_row+2, box_col:box_col+3]:
            return False
        return True
    
    def solve(grid):
        for i in range(6):
            for j in range(6):
                if grid[i, j] == 0:
                    nums = list(range(1, 7))
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
        solution = np.zeros((6, 6), dtype=np.int32)
        if solve(solution):
            # Create puzzle by removing numbers
            puzzle = solution.copy()
            positions = [(i, j) for i in range(6) for j in range(6)]
            random.shuffle(positions)
            
            # Remove 18-22 numbers (leaving 14-18 clues)
            num_to_remove = random.randint(18, 22)
            for i in range(num_to_remove):
                row, col = positions[i]
                puzzle[row, col] = 0
            
            return puzzle, solution
    
    raise Exception(f"Failed to generate valid sudoku after {max_attempts} attempts")


def augment_6x6_sudoku(puzzle, solution, n_augmentations=100):
    """Generate augmented versions of a 6×6 sudoku puzzle"""
    augmented_puzzles = []
    augmented_solutions = []
    
    for _ in range(n_augmentations):
        # Number permutation
        perm = np.random.permutation([1, 2, 3, 4, 5, 6])
        mapping = {i+1: perm[i] for i in range(6)}
        mapping[0] = 0
        
        aug_puzzle = np.vectorize(mapping.get)(puzzle).copy()
        aug_solution = np.vectorize(mapping.get)(solution).copy()
        
        # Row swaps within bands (rows 0-1, 2-3, 4-5)
        if random.random() < 0.5:
            band = random.choice([0, 2, 4])
            aug_puzzle[[band, band+1]] = aug_puzzle[[band+1, band]]
            aug_solution[[band, band+1]] = aug_solution[[band+1, band]]
        
        # Column swaps within stacks (cols 0-2 or 3-5)
        if random.random() < 0.5:
            stack = random.choice([0, 3])
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


def generate_dataset(n_base_puzzles=200, n_augmentations=50, seed=42):
    """Generate dataset with augmentations"""
    random.seed(seed)
    np.random.seed(seed)
    
    all_puzzles = []
    all_solutions = []
    
    print(f"Generating {n_base_puzzles} base 6×6 puzzles...")
    for i in range(n_base_puzzles):
        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{n_base_puzzles}")
        
        puzzle, solution = generate_6x6_sudoku()
        
        # Add augmentations
        aug_puzzles, aug_solutions = augment_6x6_sudoku(puzzle, solution, n_augmentations)
        all_puzzles.extend(aug_puzzles)
        all_solutions.extend(aug_solutions)
    
    print(f"Total dataset size: {len(all_puzzles)} examples")
    return np.array(all_puzzles), np.array(all_solutions)

def verify_sudoku(solution):
    """Verify that a solution is valid"""
    solution = solution.reshape(6, 6)
    
    # Check all numbers are 1-4
    if not np.all((solution >= 1) & (solution <= 6)):
        return False
    
    # Check rows
    for row in solution:
        if len(set(row)) != 6:
            return False
    
    # Check columns
    for col in solution.T:
        if len(set(col)) != 6:
            return False
    
    # Check 2x2 boxes
    for box_r in [0, 2, 4]:  # Box rows: 0-1, 2-3, 4-5
        for box_c in [0, 3]:  # Box cols: 0-2, 3-5
            box = solution[box_r:box_r+2, box_c:box_c+3].flatten()
            if len(set(box)) != 6:
                return False
    
    return True
