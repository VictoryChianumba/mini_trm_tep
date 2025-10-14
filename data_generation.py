import numpy as np
import random

def generate_4x4_sudoku(max_attempts=100):
    """Generate a valid 4x4 Sudoku puzzle and solution"""
    
    def is_valid(grid, row, col, num):
        # Check row
        if num in grid[row]:
            return False
        # Check column
        if num in grid[:, col]:
            return False
        # Check 2x2 box
        box_row, box_col = 2 * (row // 2), 2 * (col // 2)
        if num in grid[box_row:box_row+2, box_col:box_col+2]:
            return False
        return True
    
    def solve(grid):
        for i in range(4):
            for j in range(4):
                if grid[i, j] == 0:
                    nums = list(range(1, 5))
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
        solution = np.zeros((4, 4), dtype=np.int32)
        if solve(solution):
            # Successfully generated, now create puzzle
            puzzle = solution.copy()
            positions = [(i, j) for i in range(4) for j in range(4)]
            random.shuffle(positions)
            num_to_remove = random.randint(6, 8)
            for i in range(num_to_remove):
                row, col = positions[i]
                puzzle[row, col] = 0
            
            return puzzle, solution
    
    raise Exception(f"Failed to generate valid sudoku after {max_attempts} attempts")


def augment_sudoku(puzzle, solution, n_augmentations=100):
    """Generate augmented versions of a sudoku puzzle"""
    augmented_puzzles = []
    augmented_solutions = []
    
    for _ in range(n_augmentations):
        # Random number permutation
        perm = np.random.permutation([1, 2, 3, 4])
        mapping = {i+1: perm[i] for i in range(4)}
        mapping[0] = 0  # Keep empty cells as 0
        
        aug_puzzle = np.vectorize(mapping.get)(puzzle).copy()
        aug_solution = np.vectorize(mapping.get)(solution).copy()
        
        # Random row swap within boxes (swap rows 0↔1 or 2↔3)
        if random.random() < 0.5:
            box = random.choice([0, 2])
            aug_puzzle[[box, box+1]] = aug_puzzle[[box+1, box]]
            aug_solution[[box, box+1]] = aug_solution[[box+1, box]]
        
        # Random column swap within boxes (swap cols 0↔1 or 2↔3)
        if random.random() < 0.5:
            box = random.choice([0, 2])
            aug_puzzle[:, [box, box+1]] = aug_puzzle[:, [box+1, box]]
            aug_solution[:, [box, box+1]] = aug_solution[:, [box+1, box]]
        
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
    
    print(f"Generating {n_base_puzzles} base puzzles...")
    for i in range(n_base_puzzles):
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{n_base_puzzles}")
        
        puzzle, solution = generate_4x4_sudoku()
        
        # Add augmentations
        aug_puzzles, aug_solutions = augment_sudoku(puzzle, solution, n_augmentations)
        all_puzzles.extend(aug_puzzles)
        all_solutions.extend(aug_solutions)
    
    print(f"Total dataset size: {len(all_puzzles)} examples")
    return np.array(all_puzzles), np.array(all_solutions)


def verify_sudoku(solution):
    """Verify that a solution is valid"""
    solution = solution.reshape(4, 4)
    
    # Check all numbers are 1-4
    if not np.all((solution >= 1) & (solution <= 4)):
        return False
    
    # Check rows
    for row in solution:
        if len(set(row)) != 4:
            return False
    
    # Check columns
    for col in solution.T:
        if len(set(col)) != 4:
            return False
    
    # Check 2x2 boxes
    for box_r in [0, 2]:
        for box_c in [0, 2]:
            box = solution[box_r:box_r+2, box_c:box_c+2].flatten()
            if len(set(box)) != 4:
                return False
    
    return True


# Test the data generation
if __name__ == "__main__":
    print("Testing single puzzle generation...")
    puzzle, solution = generate_4x4_sudoku()
    
    print("\nSample puzzle:")
    print(puzzle)
    print("\nSolution:")
    print(solution)
    print(f"Solution valid: {verify_sudoku(solution.flatten())}")
    
    print("\n" + "="*50)
    print("Testing augmentation...")
    aug_puzzles, aug_solutions = augment_sudoku(puzzle, solution, n_augmentations=5)
    
    print(f"Generated {len(aug_puzzles)} augmentations")
    for i in range(min(3, len(aug_puzzles))):
        print(f"\nAugmentation {i+1}:")
        print("Puzzle:")
        print(aug_puzzles[i].reshape(4, 4))
        print("Solution:")
        print(aug_solutions[i].reshape(4, 4))
        print(f"Valid: {verify_sudoku(aug_solutions[i])}")
    
    print("\n" + "="*50)
    print("Testing full dataset generation (small version)...")
    puzzles, solutions = generate_dataset(n_base_puzzles=10, n_augmentations=10, seed=42)
    
    print(f"\nDataset shapes:")
    print(f"  Puzzles: {puzzles.shape}")
    print(f"  Solutions: {solutions.shape}")
    
    # Verify all solutions are valid
    print("\nVerifying all solutions...")
    all_valid = all(verify_sudoku(sol) for sol in solutions)
    print(f"All solutions valid: {all_valid}")
    
    print("\nData generation test complete!")