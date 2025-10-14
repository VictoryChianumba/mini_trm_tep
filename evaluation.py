# evaluation.py - Enhanced evaluation with per-step tracking

import torch
import numpy as np
from tqdm import tqdm

def evaluate_with_step_breakdown(model, dataloader, device, n_supervision_steps=16):
    """
    Evaluate model and track accuracy at each supervision step
    Returns: overall_acc, avg_steps, step_accuracies
    """
    model.eval()
    
    all_correct = []
    all_steps_taken = []
    
    # Track accuracy at each step
    step_correct = [0] * n_supervision_steps
    step_total = [0] * n_supervision_steps
    
    per_position_correct_sum = 0
    per_position_total = 0
    
    with torch.no_grad():
        for puzzles, solutions in dataloader:
            puzzles = puzzles.to(device)
            solutions = solutions.to(device)
            
            predictions, q_values, _, _ = model(
                puzzles,
                n_supervision_steps=n_supervision_steps,
                training=False
            )
            
            # Check accuracy at each step
            for step_idx, pred in enumerate(predictions):
                pred_tokens = pred.argmax(dim=-1)
                correct = (pred_tokens == solutions).all(dim=1)
                
                step_correct[step_idx] += correct.sum().item()
                step_total[step_idx] += puzzles.size(0)
            
            # Final prediction
            final_pred = predictions[-1].argmax(dim=-1)
            correct = (final_pred == solutions).all(dim=1)
            all_correct.extend(correct.cpu().numpy())
            all_steps_taken.append(len(predictions))
            
            per_position_correct = (final_pred == solutions).float()  # [B, L]
            per_position_correct_sum += per_position_correct.sum().item()
            per_position_total += final_pred.numel()  # B * L
    
    overall_acc = np.mean(all_correct)
    avg_steps = np.mean(all_steps_taken)
    
    # Calculate accuracy at each step
    step_accuracies = [step_correct[i] / step_total[i] if step_total[i] > 0 else 0 
                      for i in range(n_supervision_steps)]
    
    per_position_acc = per_position_correct_sum / per_position_total
    
    return overall_acc, avg_steps, step_accuracies, per_position_acc


def collect_example_predictions(model, dataloader, device, n_examples=5, n_supervision_steps=16):
    """
    Collect example predictions showing step-by-step evolution
    Returns: list of (puzzle, predictions_per_step, target)
    """
    model.eval()
    examples = []
    
    with torch.no_grad():
        for puzzles, solutions in dataloader:
            if len(examples) >= n_examples:
                break
            
            puzzles = puzzles.to(device)
            solutions = solutions.to(device)
            
            # Get predictions at each step
            predictions, q_values, y_states, z_states = model(
                puzzles,
                n_supervision_steps=n_supervision_steps,
                training=False
            )
            
            # Convert to numpy
            for i in range(min(puzzles.size(0), n_examples - len(examples))):
                puzzle = puzzles[i].cpu().numpy()
                target = solutions[i].cpu().numpy()
                
                # Predictions at each step
                preds_per_step = [pred[i].argmax(dim=-1).cpu().numpy() 
                                 for pred in predictions]
                
                # Q-values at each step
                q_vals = [q[i].item() for q in q_values]
                
                examples.append({
                    'puzzle': puzzle,
                    'predictions': preds_per_step,
                    'target': target,
                    'q_values': q_vals,
                    'correct': np.array_equal(preds_per_step[-1], target)
                })
    
    return examples


def evaluate_error_analysis(model, dataloader, device, n_supervision_steps=16):
    """
    Analyze what types of puzzles the model gets wrong
    """
    model.eval()
    
    errors = []
    
    with torch.no_grad():
        for puzzles, solutions in dataloader:
            puzzles = puzzles.to(device)
            solutions = solutions.to(device)
            
            predictions, _, _, _ = model(
                puzzles,
                n_supervision_steps=n_supervision_steps,
                training=False
            )
            
            final_pred = predictions[-1].argmax(dim=-1)
            
            for i in range(puzzles.size(0)):
                if not (final_pred[i] == solutions[i]).all():
                    # Count number of given clues
                    n_clues = (puzzles[i] != 0).sum().item()
                    
                    # Count wrong positions
                    n_wrong = (final_pred[i] != solutions[i]).sum().item()
                    
                    errors.append({
                        'puzzle': puzzles[i].cpu().numpy(),
                        'prediction': final_pred[i].cpu().numpy(),
                        'target': solutions[i].cpu().numpy(),
                        'n_clues': n_clues,
                        'n_wrong': n_wrong,
                    })
    
    return errors