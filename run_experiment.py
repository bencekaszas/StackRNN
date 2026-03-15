import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import linen as nn
import optax
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from functools import partial
import os

from constants import *
from data_gen import generate_rev_trace, generate_fixed_batch
from models import StackRNN

# Import constants for plotting
from constants import ACT_PUSH_0, ACT_PUSH_1, ACT_POP, STACK_NULL
from visualize import evaluate_and_visualize, plot_deepmind_style, plot_state_trajectory, plot_final_stack_distribution, plot_read_fidelity

BASE_OUTPUT_DIR = "results/confidence_study_soft"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

def create_train_state(model, key, learning_rate, dummy_input):
    params = model.init(key, dummy_input)['params']
    tx = optax.chain(
        optax.adam(learning_rate)
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def masked_loss(logits, targets, mask):
    """Masked softmax cross-entropy loss."""
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    masked_loss = loss * mask
    return masked_loss.sum() / jnp.maximum(mask.sum(), 1e-9)

@jax.jit
def train_step(state, batch):
    inputs, targets, mask = batch
    
    def loss_fn(params):
        logits, _ = state.apply_fn({'params': params}, x=inputs)
        loss = masked_loss(logits, targets, mask)
        acc = ((jnp.argmax(logits, -1) == targets) * mask).sum() / jnp.maximum(mask.sum(), 1e-9)
        return loss, acc
        
    (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    state = state.apply_gradients(grads=grads)
    return state, loss, acc

def evaluate(state, prompt, max_len=100, hard_actions=False):
    """Auto-regressive decoding with efficient state passing."""
    _, carry = state.apply_fn({'params': state.params}, x=prompt, hard_actions=hard_actions)

    decoder_input = jnp.full((prompt.shape[0], 1), VOCAB_EQ, dtype=jnp.int32)
    generated_sequence = []
    action_confidences = [] # To store max action probs
    
    cell = StackRNN.cell_cls(hard_actions=hard_actions)
    
    embed_params = state.params.get('input_embed', None)
    input_proj_params = state.params.get('input_proj', None)

    for _ in range(max_len):
        if embed_params is not None:
            decoder_emb = nn.Embed(VOCAB_SIZE, HIDDEN_DIM, name="input_embed").apply({'params': embed_params}, decoder_input)
        else:
            decoder_emb = jax.nn.one_hot(decoder_input, VOCAB_SIZE)
            decoder_emb = nn.Dense(HIDDEN_DIM, name="input_proj").apply({'params': input_proj_params}, decoder_emb)
        
        carry, (logits, action_probs) = cell.apply({'params': state.params['ScanStackRNNCell_0']}, carry, decoder_emb[:, 0])
        
        # Max prob as confidence metric (sharpness of policy)
        action_confidences.append(jnp.max(action_probs, axis=-1))
        
        next_token = jnp.argmax(logits, axis=-1)
        generated_sequence.append(next_token)
        
        if (next_token == VOCAB_EOS).all():
            break

        decoder_input = next_token[:, None]

    return jnp.concatenate(generated_sequence, axis=0), np.mean(action_confidences)

def run_full_evaluation(state, milestone_step):
    """Run a thorough OOD evaluation and save results to a milestone folder."""
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"step_{milestone_step}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n--- Running Milestone Evaluation (Step {milestone_step}) ---")
    
    # 1. Generate Visualizations (moderate and long OOD)
    # Using hard_actions=False as requested to see soft-stack evolution/drift
    VIS_L = 40
    vis_prompt = generate_fixed_batch(1, VIS_L)
    full_seq, stack_hist, action_hist, state_hist = evaluate_and_visualize(state, vis_prompt, max_len=VIS_L+10, hard_actions=False)
    plot_deepmind_style(full_seq, stack_hist, action_hist, os.path.join(output_dir, "stack_visualization.png"))
    plot_state_trajectory(state_hist, VIS_L, os.path.join(output_dir, "state_trajectory.png"))
    plot_read_fidelity(stack_hist, full_seq, VIS_L, os.path.join(output_dir, "read_fidelity.png"))

    VIS_L_LONG = 300
    vis_prompt_long = generate_fixed_batch(1, VIS_L_LONG)
    full_seq_long, stack_hist_long, action_hist_long, state_hist_long = evaluate_and_visualize(state, vis_prompt_long, max_len=VIS_L_LONG+10, hard_actions=False)
    plot_deepmind_style(full_seq_long, stack_hist_long, action_hist_long, os.path.join(output_dir, "stack_visualization_long.png"))
    plot_state_trajectory(state_hist_long, VIS_L_LONG, os.path.join(output_dir, "state_trajectory_long.png"))
    plot_read_fidelity(stack_hist_long, full_seq_long, VIS_L_LONG, os.path.join(output_dir, "read_fidelity_long.png"))
    plot_final_stack_distribution(stack_hist_long, os.path.join(output_dir, "final_stack_dist.png"))

    # 2. Accuracy Benchmarking
    TEST_LENGTHS = [10, 20, 40, 60, 100, 200, 300, 400, 500]
    final_results = {}
    
    for L in TEST_LENGTHS:
        # Fewer samples for very long sequences
        N_SAMPLES = 100 if L <= 100 else 50
        prompts = generate_fixed_batch(N_SAMPLES, L)
        correct_predictions = 0
        token_accuracies_l = []
        confidences_l = []
        
        for i in range(N_SAMPLES):
            prompt = prompts[i:i+1, :]
            prompt_bits = prompt[0, :L]
            # hard_actions=False
            generated, conf = evaluate(state, jnp.array(prompt_bits[None, :]), max_len=L+10, hard_actions=False)
            
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
            
            # Pad/Clip for token accuracy calculation
            if len(generated_output) > len(ground_truth):
                generated_output = generated_output[:len(ground_truth)]
            elif len(generated_output) < len(ground_truth):
                generated_output = np.concatenate([generated_output, [VOCAB_EOS] * (len(ground_truth) - len(generated_output))])
            
            token_accuracy = (generated_output == ground_truth).mean()
            token_accuracies_l.append(token_accuracy)
            confidences_l.append(conf)
                
        seq_acc = correct_predictions / N_SAMPLES
        token_acc = np.mean(token_accuracies_l)
        avg_conf = np.mean(confidences_l)
        print(f"Len {L}: Seq Acc: {seq_acc:.2%} | Confidence (MaxProb): {avg_conf:.4f}")
        final_results[L] = (seq_acc, token_acc, avg_conf)

    # 3. Plotting results for this milestone
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    lengths = sorted(final_results.keys())
    seq_accs = [final_results[l][0] for l in lengths]
    tok_accs = [final_results[l][1] for l in lengths]
    plt.plot(lengths, seq_accs, marker='o', label="Sequence Accuracy")
    plt.plot(lengths, tok_accs, marker='x', label="Token Accuracy")
    plt.axvline(x=60, color='gray', linestyle='--', label="Max Train Length (60)")
    plt.title(f"OOD Generalization (Step {milestone_step}, Soft Stack)", fontsize=14)
    plt.xlabel("String Length (Bits)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ood_generalization_plot.png"))
    plt.close()
    
    return final_results

if __name__ == "__main__":
    model = StackRNN(use_one_hot_emb=False)
    
    print(f"\n=== Starting Confidence Study (Soft Stack Inference, Milestones 6k, 8k, 10k) ===")
    
    # Use a fixed key
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((1, 2 * SEQ_LENGTH + 2), dtype=jnp.int32)
    state = create_train_state(model, key, LEARNING_RATE, dummy_input)

    MILESTONES = [6000, 8000, 10000]
    all_milestone_results = {}

    losses = []
    accs = []
    for step in range(STEPS + 1):
        rand_max_len = np.random.randint(10, SEQ_LENGTH + 1)
        batch = generate_rev_trace(BATCH_SIZE, rand_max_len)
        state, loss, acc = train_step(state, batch)
        losses.append(loss)
        accs.append(acc)
        
        if step % 500 == 0:
            print(f"Step {step} | Train Loss: {loss:.4f} | Train Acc: {acc:.2%}")

        if step in MILESTONES:
            all_milestone_results[step] = run_full_evaluation(state, step)

    # Save training curves
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Full Training Loss Curve")
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, "training_loss_curve.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(accs)
    plt.title("Full Training Accuracy Curve")
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, "training_accuracy_curve.png"))
    plt.close()
    
    # Final Summary Plot
    plt.figure(figsize=(10, 6))
    l300_accs = [all_milestone_results[s][300][0] for s in MILESTONES]
    l300_conf = [all_milestone_results[s][300][2] for s in MILESTONES]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(MILESTONES, l300_accs, marker='s', color='blue', label="Seq Acc (L=300)")
    ax2.plot(MILESTONES, l300_conf, marker='^', color='red', label="Avg Confidence (L=300)")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Sequence Accuracy", color='blue')
    ax2.set_ylabel("Max Prob Confidence (Action)", color='red')
    plt.title("Effect of Training Steps on Soft-Stack OOD (L=300)")
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, "confidence_study_summary.png"))
    plt.close()
