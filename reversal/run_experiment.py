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

OUTPUT_DIR = "/results/reversal/baseline_64D_state"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    
    cell = StackRNN.cell_cls(hard_actions=hard_actions)
    
    # Check if using one-hot
    embed_params = state.params.get('input_embed', None)
    input_proj_params = state.params.get('input_proj', None)

    for _ in range(max_len):
        if embed_params is not None:
            decoder_emb = nn.Embed(VOCAB_SIZE, HIDDEN_DIM, name="input_embed").apply({'params': embed_params}, decoder_input)
        else:
            decoder_emb = jax.nn.one_hot(decoder_input, VOCAB_SIZE)
            decoder_emb = nn.Dense(HIDDEN_DIM, name="input_proj").apply({'params': input_proj_params}, decoder_emb)
        
        carry, (logits, action_probs) = cell.apply({'params': state.params['ScanStackRNNCell_0']}, carry, decoder_emb[:, 0])
        
        next_token = jnp.argmax(logits, axis=-1)
        generated_sequence.append(next_token)
        
        if (next_token == VOCAB_EOS).all():
            break

        decoder_input = next_token[:, None]

    return jnp.concatenate(generated_sequence, axis=0)

if __name__ == "__main__":
    # Standard learned embedding
    model = StackRNN(use_one_hot_emb=False)
    
    TEST_LENGTHS = [10, 20, 40, 60, 100, 200, 300, 400, 500]
    TRAINING_SEQ_LEN = SEQ_LENGTH
    N_EVAL_SAMPLES = 100

    final_results = {}

    # Main loop
    print(f"\n=== Training StackRNN (64D Softmax State, tanh activation) ===")
    
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((1, 2 * TRAINING_SEQ_LEN + 2), dtype=jnp.int32)
    state = create_train_state(model, key, LEARNING_RATE, dummy_input)

    #Train Loop
    losses = []
    accs = []
    for step in range(STEPS + 1):
        rand_max_len = np.random.randint(10, TRAINING_SEQ_LEN + 1)
        batch = generate_rev_trace(BATCH_SIZE, rand_max_len)
        state, loss, acc = train_step(state, batch)
        losses.append(loss)
        accs.append(acc)
        if step % 500 == 0:
            print(f"Step {step} | Train Loss: {loss:.4f} | Train Acc: {acc:.2%}")

    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(OUTPUT_DIR, "training_loss_curve.png"))

    plt.figure(figsize=(10, 6))
    plt.plot(accs)
    plt.title("Training Accuracy Curve")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(OUTPUT_DIR, "training_accuracy_curve.png"))
    
    #evaluate OOD
    print(f"--- Evaluating OOD and Generating Visualizations ---")
    
    # --- Generate Visualizations for a moderate sequence ---
    VIS_L = 40
    vis_prompt = generate_fixed_batch(1, VIS_L)
    # Enable hard_actions for inference
    full_seq, stack_hist, action_hist, state_hist = evaluate_and_visualize(state, vis_prompt, max_len=VIS_L+10, hard_actions=False)
    plot_deepmind_style(full_seq, stack_hist, action_hist, os.path.join(OUTPUT_DIR, "stack_visualization.png"))
    plot_state_trajectory(state_hist, VIS_L, os.path.join(OUTPUT_DIR, "state_trajectory.png"))
    plot_read_fidelity(stack_hist, full_seq, VIS_L, os.path.join(OUTPUT_DIR, "read_fidelity.png"))
    print("Saved moderate sequence visualizations.")

    # --- Generate Visualizations for a very long OOD sequence ---
    VIS_L_LONG = 500
    vis_prompt_long = generate_fixed_batch(1, VIS_L_LONG)
    full_seq_long, stack_hist_long, action_hist_long, state_hist_long = evaluate_and_visualize(state, vis_prompt_long, max_len=VIS_L_LONG+10, hard_actions=False)
    plot_deepmind_style(full_seq_long, stack_hist_long, action_hist_long, os.path.join(OUTPUT_DIR, "stack_visualization_long.png"))
    plot_state_trajectory(state_hist_long, VIS_L_LONG, os.path.join(OUTPUT_DIR, "state_trajectory_long.png"))
    plot_read_fidelity(stack_hist_long, full_seq_long, VIS_L_LONG, os.path.join(OUTPUT_DIR, "read_fidelity_long.png"))
    plot_final_stack_distribution(stack_hist_long, os.path.join(OUTPUT_DIR, "final_stack_dist.png"))
    print("Saved long OOD sequence visualizations.")

    for L in TEST_LENGTHS:
        # Fewer samples for very long sequences
        N_SAMPLES = 100 if L <= 100 else 50
        prompts = generate_fixed_batch(N_SAMPLES, L)
        correct_predictions = 0
        token_accuracies_l = []
        for i in range(N_SAMPLES):
            prompt = prompts[i:i+1, :]
            prompt_bits = prompt[0, :L]
            # Enable hard_actions for inference
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
    plt.title("OOD Generalization (64D Softmax State, tanh activation)", fontsize=14)
    plt.xlabel("String Length (Bits)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ood_generalization_plot.png"))
    plt.show()
