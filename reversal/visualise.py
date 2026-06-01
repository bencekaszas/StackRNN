import jax
import jax.numpy as jnp
from flax import linen as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from constants import *
from models import StackRNN

def evaluate_and_visualize(state, prompt, max_len=100, hard_actions=False):
    """
    Runs the full encode-decode process step-by-step to collect activations.
    """
    # --- Data Collection ---
    full_sequence = [t.item() for t in prompt[0]]
    action_history = [] 
    stack_history = []
    state_history = []
    
    # 1. Initialize carry
    batch_size = prompt.shape[0]
    init_stack = jnp.zeros((batch_size, STACK_DEPTH, STACK_VOCAB_SIZE))
    init_stack = init_stack.at[:, :, STACK_NULL].set(1.0)
    init_state = jnp.zeros((batch_size, NUM_STATES), dtype=jnp.float32)
    carry = (init_stack, init_state)
    
    # 2. Get parameters and cell
    embed_params = state.params.get('input_embed', None)
    input_proj_params = state.params.get('input_proj', None)
    cell_params = state.params['ScanStackRNNCell_0']
    
    cell = StackRNN.cell_cls(hard_actions=hard_actions)

    # 3. Helper for embedding/projection
    def get_emb(x):
        if embed_params is not None:
            return nn.Embed(VOCAB_SIZE, HIDDEN_DIM, name="input_embed").apply({'params': embed_params}, x)
        else:
            x_one_hot = jax.nn.one_hot(x, VOCAB_SIZE)
            return nn.Dense(HIDDEN_DIM, name="input_proj").apply({'params': input_proj_params}, x_one_hot)

    # --- Encoding Phase ---
    prompt_emb = get_emb(prompt)
    for i in range(prompt.shape[1]):
        stack_history.append(carry[0])
        state_history.append(carry[1])
        carry, (_, action_probs) = cell.apply({'params': cell_params}, carry, prompt_emb[:, i])
        action_history.append(action_probs)

    # --- Decoding Phase ---
    decoder_input = prompt[:, -1:]
    for _ in range(max_len):
        stack_history.append(carry[0])
        state_history.append(carry[1])
        
        decoder_emb = get_emb(decoder_input)
        carry, (logits, action_probs) = cell.apply({'params': cell_params}, carry, decoder_emb[:, 0])
        action_history.append(action_probs)
        
        next_token = jnp.argmax(logits, axis=-1)
        full_sequence.append(next_token.item())
        
        if (next_token == VOCAB_EOS).all():
            break
        decoder_input = next_token[:, None]

    # Append final states
    stack_history.append(carry[0])
    state_history.append(carry[1])

    return (np.array(full_sequence),
            np.array(stack_history),
            np.array(action_history),
            np.array(state_history).squeeze())

def plot_deepmind_style(full_sequence, stack_history, action_history, file_path):
    action_history = np.array(action_history).squeeze()
    stack_history = np.array(stack_history).squeeze()
    push_strength = action_history[:, ACT_PUSH_0] + action_history[:, ACT_PUSH_1]
    pop_strength = action_history[:, ACT_POP]
    noop_strength = action_history[:, ACT_NOOP]
    bar_data = np.vstack([push_strength, pop_strength, noop_strength]).T
    num_actions = len(action_history)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [1, 2]})
    indices = np.arange(num_actions)
    ax1.bar(indices, bar_data[:num_actions, 0], 0.5, label='PUSH', color='blue')
    ax1.bar(indices, bar_data[:num_actions, 1], 0.5, bottom=bar_data[:num_actions, 0], label='POP', color='green')
    ax1.bar(indices, bar_data[:num_actions, 2], 0.5, bottom=bar_data[:num_actions, 0] + bar_data[:num_actions, 1], label='NO_OP', color='red')
    ax1.set_title('Probability of stack action, per input token', fontsize=16)
    ax1.set_xticks(indices)
    ax1.set_xticklabels([str(t) for t in full_sequence[:num_actions]])
    ax1.legend()

    stack_contents = np.argmax(stack_history, axis=-1).T
    stack_masked = np.ma.masked_where(stack_contents == STACK_NULL, stack_contents)
    ax2.imshow(stack_masked, aspect='auto', cmap='viridis', interpolation='nearest')
    ax2.set_title('Stack evolution for an input sequence', fontsize=16)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def plot_state_trajectory(state_history, prompt_len, file_path):
    T, D = state_history.shape
    fig, axes = plt.subplots(D, 1, figsize=(12, 4 * D), sharex=True)
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

def plot_read_fidelity(stack_history, full_sequence, prompt_len, file_path):
    stack_history = np.array(stack_history).squeeze()
    decoding_stacks = stack_history[prompt_len+1:] 
    top_of_stack = decoding_stacks[:, 0, :] 
    
    # Expected bits (prompt reversed)
    expected_bits = full_sequence[:prompt_len-1][::-1]
    expected_stack_vals = []
    for b in expected_bits:
        if b == VOCAB_0: expected_stack_vals.append(STACK_0)
        elif b == VOCAB_1: expected_stack_vals.append(STACK_1)
    
    T_compare = min(len(top_of_stack), len(expected_stack_vals))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.imshow(top_of_stack[:T_compare].T, aspect='auto', cmap='viridis')
    ax1.set_title('Top-of-Stack Distribution (Depth 0) during Decoding')
    ax1.set_yticks([STACK_NULL, STACK_0, STACK_1])
    ax1.set_yticklabels(['NULL', '0', '1'])
    
    actual_bits = np.argmax(top_of_stack[:T_compare], axis=-1)
    ax2.step(range(T_compare), actual_bits, where='post', label='Read from Stack', color='blue', linewidth=2)
    ax2.step(range(T_compare), expected_stack_vals[:T_compare], where='post', label='Ground Truth (Reversed)', color='red', linestyle='--', alpha=0.7)
    
    ax2.set_title('Read Fidelity: Actual vs Expected Bit')
    ax2.set_ylabel('Stack Label (0=NULL, 1=Bit0, 2=Bit1)')
    ax2.set_xlabel('Decoding Step')
    ax2.set_yticks([STACK_NULL, STACK_0, STACK_1])
    ax2.set_yticklabels(['NULL', '0', '1'])
    ax2.set_ylim(-0.5, 2.5) 
    ax2.legend()
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def plot_final_stack_distribution(stack_history, file_path, last_n_steps=15):
    """
    Optimized version of stack distribution plotting.
    Uses a single axis and horizontal bars to avoid the overhead of many subplots.
    """
    stack_history = np.array(stack_history).squeeze()
    stack_portion = stack_history[-last_n_steps:, :, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        stack_portion = np.nan_to_num(stack_portion / stack_portion.sum(axis=-1, keepdims=True))
    T, D, V = stack_portion.shape
    fig, ax = plt.subplots(figsize=(T * 0.4, D * 0.2))
    color_0, color_1, color_null = '#440154', '#fde725', 'lightgray'
    for t in range(T):
        for d in range(D):
            dist = stack_portion[t, d, :]
            y_base = D - d
            x_base = t
            ax.barh(y_base, dist[STACK_0], left=x_base, height=0.8, color=color_0)
            ax.barh(y_base, dist[STACK_1], left=x_base + dist[STACK_0], height=0.8, color=color_1)
            ax.barh(y_base, dist[STACK_NULL], left=x_base + dist[STACK_0] + dist[STACK_1], height=0.8, color=color_null)
    ax.set_xlim(0, T)
    ax.set_ylim(0.5, D + 0.5)
    ax.set_axis_off()
    ax.set_title(f'Final Stack Distributions (Last {T} steps)', fontsize=16)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
