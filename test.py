from sudoku_6x6_generator import generate_6x6_sudoku, verify_sudoku

puzzle, solution = generate_6x6_sudoku()
print("Puzzle:")
print(puzzle)
print("\nSolution:")
print(solution)
print(f"\nSolution valid: {verify_sudoku(solution)}")

# Check box structure (2×3)
print("\nBoxes:")
for box_r in [0, 2, 4]:
    for box_c in [0, 3]:
        box = solution[box_r:box_r+2, box_c:box_c+3].flatten()
        print(f"Box ({box_r},{box_c}): {sorted(box)} - Valid: {len(set(box)) == 6}")