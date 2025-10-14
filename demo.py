# demo.py - Interactive demo generator

import json

def generate_html_demo(model_path, examples, output_path='demo.html'):
    """
    Generate a standalone HTML file with interactive Sudoku solver
    """
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>TRM Sudoku Solver Demo</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .container {
            display: flex;
            gap: 40px;
            margin-top: 30px;
        }
        .sudoku-grid {
            display: grid;
            grid-template-columns: repeat(4, 60px);
            grid-template-rows: repeat(4, 60px);
            gap: 2px;
            background-color: #000;
            border: 3px solid #000;
            padding: 0;
        }
        .cell {
            background-color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            font-weight: bold;
            cursor: pointer;
        }
        .cell.given {
            background-color: #e3f2fd;
            color: #1976d2;
        }
        .cell.predicted {
            background-color: #fff3e0;
            color: #f57c00;
        }
        .cell.correct {
            background-color: #e8f5e9;
            color: #388e3c;
        }
        .cell.wrong {
            background-color: #ffebee;
            color: #d32f2f;
        }
        .controls {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        button {
            padding: 12px 24px;
            font-size: 16px;
            cursor: pointer;
            border: none;
            border-radius: 4px;
            background-color: #1976d2;
            color: white;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #1565c0;
        }
        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .step-info {
            margin-top: 20px;
            padding: 15px;
            background-color: white;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .examples {
            margin-top: 40px;
        }
        .example-grid {
            display: inline-block;
            margin: 10px;
        }
        select {
            padding: 10px;
            font-size: 16px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <h1>🧩 Tiny Recursive Model - 4x4 Sudoku Solver</h1>
    
    <div class="container">
        <div>
            <h2>Input Puzzle</h2>
            <div id="puzzle" class="sudoku-grid"></div>
            
            <div class="controls">
                <select id="exampleSelect">
                    <option value="">Load an example...</option>
                    {example_options}
                </select>
                <button onclick="solvePuzzle()">Solve Step-by-Step</button>
                <button onclick="reset()">Reset</button>
            </div>
            
            <div class="step-info">
                <p><strong>Current Step:</strong> <span id="currentStep">0</span></p>
                <p><strong>Confidence (Q-value):</strong> <span id="qValue">-</span></p>
                <p><strong>Status:</strong> <span id="status">Ready</span></p>
            </div>
        </div>
        
        <div>
            <h2>Model Prediction</h2>
            <div id="solution" class="sudoku-grid"></div>
        </div>
    </div>
    
    <div class="examples">
        <h2>Example Puzzles</h2>
        <p>Click on any example to load it:</p>
        <div id="examplesList"></div>
    </div>
    
    <script>
        const examples = {examples_json};
        
        let currentPuzzle = null;
        let currentStep = 0;
        let solving = false;
        
        function createGrid(containerId) {
            const container = document.getElementById(containerId);
            container.innerHTML = '';
            for (let i = 0; i < 16; i++) {
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.id = `${containerId}-${i}`;
                container.appendChild(cell);
            }
        }
        
        function displayPuzzle(puzzle) {
            for (let i = 0; i < 16; i++) {
                const cell = document.getElementById(`puzzle-${i}`);
                const value = puzzle[i];
                cell.textContent = value === 0 ? '' : value;
                cell.className = value === 0 ? 'cell' : 'cell given';
            }
        }
        
        function displaySolution(solution, target = null) {
            for (let i = 0; i < 16; i++) {
                const cell = document.getElementById(`solution-${i}`);
                const value = solution[i];
                cell.textContent = value === 0 ? '' : value;
                
                if (target) {
                    if (value === target[i]) {
                        cell.className = 'cell correct';
                    } else if (value !== 0) {
                        cell.className = 'cell wrong';
                    } else {
                        cell.className = 'cell';
                    }
                } else {
                    cell.className = value === 0 ? 'cell' : 'cell predicted';
                }
            }
        }
        
        function loadExample(index) {
            const example = examples[index];
            currentPuzzle = example;
            currentStep = 0;
            
            displayPuzzle(example.puzzle);
            displaySolution(Array(16).fill(0));
            
            document.getElementById('currentStep').textContent = '0';
            document.getElementById('qValue').textContent = '-';
            document.getElementById('status').textContent = 'Loaded';
        }
        
        async function solvePuzzle() {
            if (!currentPuzzle || solving) return;
            
            solving = true;
            const predictions = currentPuzzle.predictions;
            const qValues = currentPuzzle.q_values;
            const target = currentPuzzle.target;
            
            for (let step = 0; step < predictions.length; step++) {
                currentStep = step;
                document.getElementById('currentStep').textContent = step;
                document.getElementById('qValue').textContent = qValues[step].toFixed(3);
                document.getElementById('status').textContent = 'Solving...';
                
                displaySolution(predictions[step], target);
                
                await new Promise(resolve => setTimeout(resolve, 800));
            }
            
            const isCorrect = predictions[predictions.length-1].every((v, i) => v === target[i]);
            document.getElementById('status').textContent = isCorrect ? '✓ Correct!' : '✗ Incorrect';
            solving = false;
        }
        
        function reset() {
            if (currentPuzzle) {
                loadExample(examples.indexOf(currentPuzzle));
            }
        }
        
        function populateExamples() {
            const select = document.getElementById('exampleSelect');
            const list = document.getElementById('examplesList');
            
            examples.forEach((ex, i) => {
                const option = document.createElement('option');
                option.value = i;
                option.textContent = `Example ${i + 1} (${ex.correct ? 'Correct' : 'Wrong'})`;
                select.appendChild(option);
                
                // Create small preview
                const preview = document.createElement('div');
                preview.className = 'example-grid';
                preview.innerHTML = `<small>Example ${i + 1}</small>`;
                preview.onclick = () => {
                    select.value = i;
                    loadExample(i);
                };
                preview.style.cursor = 'pointer';
                list.appendChild(preview);
            });
            
            select.onchange = (e) => {
                if (e.target.value !== '') {
                    loadExample(parseInt(e.target.value));
                }
            };
        }
        
        // Initialize
        createGrid('puzzle');
        createGrid('solution');
        populateExamples();
        
        if (examples.length > 0) {
            loadExample(0);
        }
    </script>
</body>
</html>
    """
    
    # Convert examples to JSON
    examples_json = json.dumps([{
        'puzzle': ex['puzzle'].tolist(),
        'predictions': [p.tolist() for p in ex['predictions']],
        'target': ex['target'].tolist(),
        'q_values': ex['q_values'],
        'correct': ex['correct']
    } for ex in examples])
    
    example_options = '\n'.join([
        f'<option value="{i}">Example {i+1} ({"Correct" if ex["correct"] else "Wrong"})</option>'
        for i, ex in enumerate(examples)
    ])
    
    html = html_template.format(
        examples_json=examples_json,
        example_options=example_options
    )
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Generated interactive demo: {output_path}")
    print("Open it in a web browser to try the solver!")