import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import linen as nn
import optax
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import sys

# Local imports
from constants import *
from data_gen import generate_rev_trace, generate_fixed_batch
from models import StackRNN
from visualise import evaluate_and_visualize, plot_deepmind_style, plot_state_trajectory, plot_final_stack_distribution, plot_read_fidelity, plot_epsilon_analysis, plot_fidelity_vs_theory, plot_stack_entropy

# Import evaluation function
from run_experiment import evaluate

STOP_POINTS = [8000, 10000, 12000, 14000]
BASE_OUTPUT_DIR = "../results/reversal/early_stops_Q2"

def create_train_state(model, key, learning_rate, dummy_input):
    params = model.init(key, dummy_input)['params']
    tx = optax.chain(optax.adam(learning_rate))
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch):
    inputs, targets, mask = batch
    def loss_fn(params):
        logits, _ = state.apply_fn({'params': params}, x=inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        loss = (loss * mask).sum() / jnp.maximum(mask.sum(), 1e-9)
        acc = ((jnp.argmax(logits, -1) == targets) * mask).sum() / jnp.maximum(mask.sum(), 1e-9)
        return loss, acc
    (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), loss, acc

def run_full_eval(state, step):
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"step_{step}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Running Full Evaluation for Step {step} ---")
    
    # 1. Visualizations (Moderate Length)
    VIS_L = 40
    vis_prompt = generate_fixed_batch(1, VIS_L)
    full_seq, stack_hist, action_hist, state_hist, buffer_hist = evaluate_and_visualize(state, vis_prompt, max_len=VIS_L+10, hard_actions=False)
    plot_deepmind_style(full_seq, stack_hist, action_hist, os.path.join(output_dir, "stack_visualization.pdf"))
    plot_state_trajectory(state_hist, VIS_L, os.path.join(output_dir, "state_trajectory.pdf"))
    plot_read_fidelity(stack_hist, full_seq, VIS_L, os.path.join(output_dir, "read_fidelity.pdf"))
    plot_epsilon_analysis(full_seq, action_hist, VIS_L, os.path.join(output_dir, "epsilon_dist.pdf"), os.path.join(output_dir, "epsilon_time.pdf"), os.path.join(output_dir, "epsilon_raw.npy"))
    plot_fidelity_vs_theory(full_seq, buffer_hist, action_hist, VIS_L, os.path.join(output_dir, "fidelity_vs_theory.pdf"))
    plot_stack_entropy(stack_hist, VIS_L, os.path.join(output_dir, "stack_entropy.pdf"))

    # 2. Visualizations (Long Length)
    VIS_L_LONG = 500
    vis_prompt_long = generate_fixed_batch(1, VIS_L_LONG)
    full_seq_long, stack_hist_long, action_hist_long, state_hist_long, buffer_hist_long = evaluate_and_visualize(state, vis_prompt_long, max_len=VIS_L_LONG+10, hard_actions=False)
    plot_deepmind_style(full_seq_long, stack_hist_long, action_hist_long, os.path.join(output_dir, "stack_visualization_long.pdf"))
    plot_state_trajectory(state_hist_long, VIS_L_LONG, os.path.join(output_dir, "state_trajectory_long.pdf"))
    plot_read_fidelity(stack_hist_long, full_seq_long, VIS_L_LONG, os.path.join(output_dir, "read_fidelity_long.pdf"))
    plot_final_stack_distribution(stack_hist_long, os.path.join(output_dir, "final_stack_dist.pdf"))
    plot_epsilon_analysis(full_seq_long, action_hist_long, VIS_L_LONG, os.path.join(output_dir, "epsilon_dist_long.pdf"), os.path.join(output_dir, "epsilon_time_long.pdf"), os.path.join(output_dir, "epsilon_raw_long.npy"))
    plot_fidelity_vs_theory(full_seq_long, buffer_hist_long, action_hist_long, VIS_L_LONG, os.path.join(output_dir, "fidelity_vs_theory_long.pdf"))
    plot_stack_entropy(stack_hist_long, VIS_L_LONG, os.path.join(output_dir, "stack_entropy_long.pdf"))
    
    # 3. OOD Generalization Plot (matching run_experiment.py exactly)
    TEST_LENGTHS = [10, 20, 40, 60, 100, 200, 300, 400, 500]
    final_results = {}
    for L in TEST_LENGTHS:
        N_SAMPLES = 100 if L <= 100 else 50
        prompts = generate_fixed_batch(N_SAMPLES, L)
        correct_predictions = 0
        token_accuracies_l = []
        for i in range(N_SAMPLES):
            prompt = prompts[i:i+1, :]
            prompt_bits = prompt[0, :L]
            # Match run_experiment.py choice of hard_actions (False in the loop provided)
            generated = evaluate(state, jnp.array(prompt_bits[None, :]), max_len=L+10, hard_actions=False)
            
            generated_output = generated
            ground_truth = np.asarray(prompt_bits[::-1])
            ground_truth = np.concatenate([ground_truth, [VOCAB_EOS]])
            
            is_correct = False
            if len(generated_output) >= len(ground_truth):
                if np.array_equal(generated_output[:len(ground_truth)], ground_truth):
                    if len(generated_output) == len(ground_truth) or generated_output[len(ground_truth)] == VOCAB_EOS:
                        is_correct = True
            if is_correct:
                correct_predictions += 1
            
            if len(generated_output) > len(ground_truth):
                generated_output = generated_output[:len(ground_truth)]
            elif len(generated_output) < len(ground_truth):
                generated_output = np.concatenate([generated_output, [VOCAB_EOS] * (len(ground_truth) - len(generated_output))])
            token_accuracy = (generated_output == ground_truth).mean()
            token_accuracies_l.append(token_accuracy)
                
        seq_acc = correct_predictions / N_SAMPLES
        token_acc = np.mean(token_accuracies_l)
        print(f"Len {L}: Seq Acc: {seq_acc:.2%} | Token Acc: {token_acc:.2%}")
        final_results[L] = (seq_acc, token_acc)

    # --- Plotting OOD Accuracy ---
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    lengths = sorted(final_results.keys())
    seq_accs = [final_results[l][0] for l in lengths]
    tok_accs = [final_results[l][1] for l in lengths]
    plt.plot(lengths, seq_accs, marker='o', label="Sequence Accuracy")
    plt.plot(lengths, tok_accs, marker='x', label="Token Accuracy")
    plt.axvline(x=60, color='gray', linestyle='--', label="Max Train Length (60)")
    plt.title(f"OOD Generalization (Step {step}, Q=2)", fontsize=14)
    plt.xlabel("String Length (Bits)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ood_generalization_plot.pdf"))
    plt.close()

if __name__ == "__main__":
    model = StackRNN(use_one_hot_emb=False)
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((1, 2 * SEQ_LENGTH + 2), dtype=jnp.int32)
    state = create_train_state(model, key, LEARNING_RATE, dummy_input)
    
    losses, accs = [], []
    for step in range(max(STOP_POINTS) + 1):
        rand_max_len = np.random.randint(10, SEQ_LENGTH + 1)
        batch = generate_rev_trace(64, rand_max_len)
        state, loss, acc = train_step(state, batch)
        losses.append(loss)
        accs.append(acc)
        
        if step in STOP_POINTS:
            print(f"\nBreakpoint Checkpoint at step {step}!")
            run_full_eval(state, step)
            
        if step % 500 == 0:
            print(f"Step {step} | Loss: {loss:.4f} | Acc: {acc:.2%}")
            
    # Final training plots
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss (Q=2)")
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, "training_loss.pdf"))
    plt.close()
    
    plt.figure()
    plt.plot(accs)
    plt.title("Training Accuracy (Q=2)")
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, "training_accuracy.pdf"))
    plt.close()
