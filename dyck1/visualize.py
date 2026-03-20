import jax
import jax.numpy as jnp
from flax import linen as nn
import matplotlib.pyplot as plt
import numpy as np
from dyck_constants import *
from dyck_models import DyckStackRNN, DyckStackRNNCell

def evaluate_and_visualize_dyck(state, prompt, max_len=100, hard_actions=False):
    """
    Runs the full encode-decode process step-by-step to collect activations for Dyck-1.
    """
    full_sequence = [t.item() for t in prompt[0]]
    action_history = [] 
    stack_history = []
    state_history = []
    
    batch_size = prompt.shape[0]
    init_stack = jnp.zeros((batch_size, STACK_DEPTH, DYCK_STACK_VOCAB_SIZE))
    init_stack = init_stack.at[:, :, DYCK_STACK_NULL].set(1.0)
    init_state = jnp.zeros((batch_size, DYCK_NUM_STATES), dtype=jnp.float32)
    carry = (init_stack, init_state)
    
    embed_params = state.params.get('input_embed', None)
    cell_key = [k for k in state.params.keys() if 'Cell' in k][0]
    cell_params = state.params[cell_key]
    cell = DyckStackRNNCell(hard_actions=hard_actions)

    def get_emb(x):
        return nn.Embed(DYCK_VOCAB_SIZE, HIDDEN_DIM, name="input_embed").apply({'params': embed_params}, x)

    # --- Encoding Phase ---
    prompt_emb = get_emb(prompt)
    for i in range(prompt.shape[1]):
        stack_history.append(carry[0])
        state_history.append(carry[1])
        carry, (_, action_probs) = cell.apply({'params': cell_params}, carry, prompt_emb[:, i])
        action_history.append(action_probs)

    # --- Decoding Phase ---
    decoder_input = jnp.full((batch_size, 1), DYCK_EQ, dtype=jnp.int32)
    full_sequence.append(DYCK_EQ)
    
    for _ in range(max_len):
        stack_history.append(carry[0])
        state_history.append(carry[1])
        
        decoder_emb = get_emb(decoder_input)
        carry, (logits, action_probs) = cell.apply({'params': cell_params}, carry, decoder_emb[:, 0])
        action_history.append(action_probs)
        
        next_token = jnp.argmax(logits, axis=-1)
        full_sequence.append(next_token.item())
        
        if (next_token == DYCK_EOS).all():
            break
        decoder_input = next_token[:, None]

    # Final state (after last token/EOS)
    stack_history.append(carry[0])
    state_history.append(carry[1])

    return (np.array(full_sequence),
            np.array(stack_history),
            np.array(action_history),
            np.array(state_history).squeeze())

def plot_dyck_stack_viz(full_sequence, stack_history, action_history, file_path):
    action_history = np.array(action_history).squeeze()
    stack_history = np.array(stack_history).squeeze()
    
    # Stack Action Probabilities
    push_strength = action_history[:, ACT_PUSH_0] + action_history[:, ACT_PUSH_1]
    pop_strength = action_history[:, ACT_POP]
    noop_strength = action_history[:, ACT_NOOP]
    bar_data = np.vstack([push_strength, pop_strength, noop_strength]).T
    num_actions = len(action_history)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [1, 2]})
    
    indices = np.arange(num_actions)
    ax1.bar(indices, bar_data[:, 0], 0.5, label='PUSH', color='blue')
    ax1.bar(indices, bar_data[:, 1], 0.5, bottom=bar_data[:, 0], label='POP', color='green')
    ax1.bar(indices, bar_data[:, 2], 0.5, bottom=bar_data[:, 0] + bar_data[:, 1], label='NO_OP', color='red')
    ax1.set_title('Probability of stack action, per input token', fontsize=16)
    ax1.legend(loc='upper right')
    ax1.set_xlim(-0.5, num_actions - 0.5)

    # Stack Contents (Probability of OPEN)
    # Aligning stack_history[t] with the token processed at time t
    # stack_history[t] is the stack state BEFORE the action at time t
    stack_probs = stack_history[:num_actions, :, DYCK_STACK_OPEN].T
    
    # Find active depth for cropping
    threshold = 0.05
    active_cells = np.where(stack_probs > threshold)
    if len(active_cells[0]) > 0:
        max_active_depth = np.max(active_cells[0])
    else:
        max_active_depth = 5
    crop_depth = min(max_active_depth + 5, stack_probs.shape[0])
    stack_cropped = stack_probs[:crop_depth, :]
    
    # Plot heatmap
    # Using origin='lower' so depth 0 is at the "bottom" relative to higher indices
    # But wait, in our soft stack, index 0 is the TOP.
    # To make it grow UPWARDS, index 0 (top) should be at the top of the colored area.
    # If we use origin='lower', index 0 is at the bottom. PUSHing shifts 0->1, 1->2.
    # So the element at 0 moves UP to 1. This looks like growing!
    im = ax2.imshow(stack_cropped, aspect='auto', cmap='viridis', interpolation='nearest', origin='lower')
    ax2.set_title('Stack evolution (P(Open))', fontsize=16)
    ax2.set_ylabel('Stack Depth')
    
    # Add tokens above the heatmap
    token_labels = []
    for t in full_sequence[:num_actions]:
        if t == DYCK_OPEN: token_labels.append('(')
        elif t == DYCK_CLOSE: token_labels.append(')')
        elif t == DYCK_EQ: token_labels.append('=')
        elif t == DYCK_EOS: token_labels.append('EOS')
        else: token_labels.append(str(t))
    
    ax2.set_xticks(indices)
    ax2.set_xticklabels(token_labels, fontsize=12)
    ax2.xaxis.tick_top() # Put labels on top of the heatmap
    
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def plot_dyck_ood_generalization(results, title, file_path, max_train_len=60):
    import seaborn as sns
    sns.set_style("whitegrid")
    lengths = sorted(results.keys())
    seq_accs = [results[l][0] for l in lengths]
    tok_accs = [results[l][1] for l in lengths]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, seq_accs, marker='o', label="Sequence Accuracy")
    plt.plot(lengths, tok_accs, marker='x', label="Token Accuracy")
    plt.axvline(x=max_train_len, color='gray', linestyle='--', label=f"Max Train Length ({max_train_len})")
    plt.title(title, fontsize=14)
    plt.xlabel("Prefix Length", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def plot_dyck_state_trajectory(state_history, prompt_len, file_path):
    T, D = state_history.shape
    fig, axes = plt.subplots(D, 1, figsize=(12, 2 * D), sharex=True)
    if D == 1: axes = [axes]
    for i in range(D):
        axes[i].plot(state_history[:, i], label=f'State Dim {i}', color=f'C{i}', linewidth=2)
        axes[i].axvline(x=prompt_len, color='gray', linestyle='--')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.3)
    fig.suptitle('Controller State Components over Time', fontsize=16)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
