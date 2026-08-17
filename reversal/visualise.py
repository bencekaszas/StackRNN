import jax
import jax.numpy as jnp
from flax import linen as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from constants import *
from models import StackRNN
import seaborn as sns

def evaluate_and_visualize(state, prompt, max_len=100, hard_actions=False, num_states=NUM_STATES):
    """
    Runs the full encode-decode process step-by-step to collect activations.
    """
    # --- Data Collection ---
    full_sequence = [t.item() for t in prompt[0]]
    action_history = [] 
    stack_history = []
    state_history = []
    buffer_history = []
    
    # 1. Initialize carry
    batch_size = prompt.shape[0]
    init_stack = jnp.zeros((batch_size, STACK_DEPTH, STACK_VOCAB_SIZE))
    init_stack = init_stack.at[:, :, STACK_NULL].set(1.0)
    init_state = jnp.zeros((batch_size, num_states), dtype=jnp.float32)
    carry = (init_stack, init_state)
    
    # 2. Get parameters and cell
    embed_params = state.params.get('input_embed', None)
    input_proj_params = state.params.get('input_proj', None)
    cell_params = state.params['ScanStackRNNCell_0']
    
    cell = StackRNN.cell_cls(hard_actions=hard_actions, num_states=num_states)

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
        carry, (logits, action_probs) = cell.apply({'params': cell_params}, carry, prompt_emb[:, i])
        action_history.append(action_probs)
        buffer_history.append(jax.nn.softmax(logits))

    # --- Decoding Phase ---
    decoder_input = prompt[:, -1:]
    for _ in range(max_len):
        stack_history.append(carry[0])
        state_history.append(carry[1])
        
        decoder_emb = get_emb(decoder_input)
        carry, (logits, action_probs) = cell.apply({'params': cell_params}, carry, decoder_emb[:, 0])
        action_history.append(action_probs)
        buffer_probs = jax.nn.softmax(logits)
        buffer_history.append(buffer_probs)
        
        next_token = jnp.argmax(logits, axis=-1)
        full_sequence.append(next_token.item())
        
        if (next_token == VOCAB_EOS).all():
            break
        decoder_input = next_token[:, None]

    return (np.array(full_sequence),
            np.array(stack_history),
            np.array(action_history),
            np.array(state_history).squeeze(),
            np.array(buffer_history).squeeze())

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
    
    token_labels = []
    for t in full_sequence[:num_actions]:
        if t == VOCAB_0: token_labels.append('0')
        elif t == VOCAB_1: token_labels.append('1')
        elif t == VOCAB_EQ: token_labels.append('=')
        elif t == VOCAB_EOS: token_labels.append('EOS')
        else: token_labels.append('')
        
    ax1.set_xticks(indices)
    ax1.set_xticklabels(token_labels)
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
    decoding_stacks = stack_history[prompt_len:] 
    top_of_stack = decoding_stacks[:, 0, :] 
    
    # Expected bits (prompt reversed)
    expected_bits = full_sequence[:prompt_len][::-1]
    expected_stack_vals = []
    for b in expected_bits:
        if b == VOCAB_0: expected_stack_vals.append(STACK_0)
        elif b == VOCAB_1: expected_stack_vals.append(STACK_1)
    expected_stack_vals.append(STACK_NULL) 
    
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
            ax.barh(y_base, dist[STACK_0], left=t, height=0.8, color=color_0)
            ax.barh(y_base, dist[STACK_1], left=t + dist[STACK_0], height=0.8, color=color_1)
            ax.barh(y_base, dist[STACK_NULL], left=t + dist[STACK_0] + dist[STACK_1], height=0.8, color=color_null)
    ax.set_xlim(0, T)
    ax.set_ylim(0.5, D + 0.5)
    ax.set_axis_off()
    ax.set_title(f'Final Stack Distributions (Last {T} steps)', fontsize=16)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def plot_epsilon_analysis(full_sequence, action_history, prompt_len, file_path_dist, file_path_time, data_path_raw):
    """
    Improved Epsilon Analysis.
    """
    action_history = np.array(action_history).squeeze()
    epsilons = []
    
    # 1. Encoding phase (bits)
    for t in range(prompt_len):
        bit = full_sequence[t]
        target_act = ACT_PUSH_0 if bit == VOCAB_0 else ACT_PUSH_1
        epsilons.append(1.0 - action_history[t, target_act])
        
    # 2. Delimiter (EQ)
    if prompt_len < len(action_history):
        epsilons.append(1.0 - action_history[prompt_len, ACT_POP])
    
    # 3. Decoding phase (reversed bits)
    for t in range(prompt_len):
        idx = prompt_len + 1 + t
        if idx < len(action_history):
            epsilons.append(1.0 - action_history[idx, ACT_POP])
            
    epsilons = np.array(epsilons, dtype=np.float64)
    
    # Save raw data to avoid floating point interpretation issues
    np.save(data_path_raw, epsilons)
    
    # --- Distribution Plot ---
    plt.figure(figsize=(10, 6))
    sns.histplot(epsilons, kde=True, color='C0')
    plt.title('Overall Distribution of Action Errors (epsilon)', fontsize=16)
    plt.xlabel('epsilon (1 - P(target_action))')
    plt.ylabel('Frequency')
    plt.savefig(file_path_dist)
    plt.close()

    # --- Time Series Plot ---
    plt.figure(figsize=(15, 6))
    plt.plot(epsilons, 'o-', color='C0', alpha=0.7, markersize=4)
    plt.title('Epsilon over Time Steps', fontsize=16)
    plt.xlabel('Time Step')
    plt.ylabel('epsilon')
    
    # Add token labels as x-ticks
    num_steps = len(epsilons)
    token_labels = []
    for t in range(prompt_len):
        bit = full_sequence[t]
        token_labels.append('0' if bit == VOCAB_0 else '1')
    token_labels.append('=')
    for t in range(prompt_len):
        token_labels.append('R') # Reversed bit
        
    plt.xticks(range(num_steps), token_labels[:num_steps])
    
    if prompt_len > 60: # Only for long sequences
        plt.axvline(x=SEQ_LENGTH, color='red', linestyle='--', label=f'Training Horizon ({SEQ_LENGTH})')
        plt.legend()
        
    plt.grid(True, alpha=0.3)
    plt.savefig(file_path_time)
    plt.close()

def plot_fidelity_vs_theory(full_sequence, buffer_history, action_history, prompt_len, file_path):
    """
    Plots the empirical probability of the correct symbol during decoding 
    against the theoretical upper bound O(1/sqrt(n*epsilon)).
    """
    buffer_history = np.array(buffer_history).squeeze()
    action_history = np.array(action_history).squeeze()
    
    # bits to reverse
    bits = full_sequence[:prompt_len]
    expected_reversed = bits[::-1]
    
    # 1. Estimate average epsilon (error rate) from action history
    epsilons = []
    for t in range(prompt_len):
        bit = full_sequence[t]
        target_act = ACT_PUSH_0 if bit == VOCAB_0 else ACT_PUSH_1
        epsilons.append(1.0 - action_history[t, target_act])
        
    for t in range(prompt_len):
        idx = prompt_len + 1 + t
        if idx < len(action_history):
            epsilons.append(1.0 - action_history[idx, ACT_POP])
    
    avg_epsilon = np.mean(epsilons)
    
    # 2. Extract Empirical Fidelity
    empirical_p = []
    n_values = np.arange(1, prompt_len + 1)
    
    for i, n in enumerate(n_values):
        idx = prompt_len + 1 + i 
        target_tok = expected_reversed[i]
        if idx < len(buffer_history):
            p_correct = buffer_history[idx, target_tok]
        else:
            p_correct = 0.0 # Model emitted EOS prematurely
        empirical_p.append(p_correct)
        
    # 3. Calculate Theoretical Bound
    theo_bound = 0.5 + 0.5 * (1.0 / np.sqrt(2 * np.pi * n_values * avg_epsilon * (1.0 - avg_epsilon)))
    theo_bound = np.clip(theo_bound, 0, 1.0)

    plt.figure(figsize=(10, 6))
    plt.plot(n_values, empirical_p, 'r-', linewidth=2, label='Empirical $P(target)$')
    plt.plot(n_values, theo_bound, 'b--', linewidth=2, label=f'Theoretical Bound ($\epsilon \\approx {avg_epsilon:.4f}$)')
    
    plt.title(f'Memory Fidelity Decay (L={prompt_len})', fontsize=16)
    plt.xlabel('Distance from phase switch ($n$)')
    plt.ylabel('Probability of correct symbol')
    plt.ylim(0.4, 1.05)
    plt.axhline(y=0.5, color='gray', linestyle=':', label='Random Guessing')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(file_path)
    plt.close()


def plot_stack_entropy(stack_history, prompt_len, file_path):
    """
    Plots the Shannon entropy of the stack during decoding.
    This serves as a measure of stack blurring over time.
    """
    stack_history = np.array(stack_history).squeeze()
    
    # Decoding starts at prompt_len + 1 (after encoding and delimiter)
    decoding_start = prompt_len + 1
    if decoding_start >= len(stack_history):
        return # Sequence too short or model stopped early
        
    decoding_stacks = stack_history[decoding_start:]
    
    top_entropies = []
    
    for t in range(len(decoding_stacks)):
        # Top of stack
        p_top = decoding_stacks[t, 0, :]
        p_safe_top = p_top[p_top > 1e-9]
        h_top = -np.sum(p_safe_top * np.log2(p_safe_top))
        top_entropies.append(h_top)
        
    n_values = np.arange(1, len(top_entropies) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, top_entropies, 'g-', linewidth=2, label='Top-of-Stack Entropy')
    
    plt.title(f'Stack Blurring During Decoding (L={prompt_len})', fontsize=16)
    plt.xlabel('Decoding Step (Distance from phase switch)')
    plt.ylabel('Entropy (bits)')
    
    # Add max entropy bound for reference
    max_entropy = np.log2(decoding_stacks.shape[-1])
    plt.axhline(y=max_entropy, color='gray', linestyle=':', label=f'Max Depth-0 Entropy ({max_entropy:.2f})')
    
    if prompt_len > 60:
        plt.axvline(x=SEQ_LENGTH, color='red', linestyle='--', label=f'Training Horizon ({SEQ_LENGTH})')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(file_path)
    plt.close()


def plot_sequence_entropy(stack_history, prompt_len, file_path_top, file_path_total):
    """
    Plots the top-of-stack and total stack entropy across the ENTIRE sequence
    (encoding phase + delimiter + decoding phase) against token number.
    """
    stack_history = np.array(stack_history).squeeze()
    
    top_entropies = []
    total_entropies = []
    
    # Iterate over every timestep
    for t in range(len(stack_history)):
        # Top of stack
        p_top = stack_history[t, 0, :]
        p_safe_top = p_top[p_top > 1e-9]
        h_top = -np.sum(p_safe_top * np.log2(p_safe_top))
        top_entropies.append(h_top)
        
        # Total stack
        p_all = stack_history[t]
        p_safe_all = np.where(p_all > 1e-9, p_all, 1.0) # log(1) = 0
        h_all = -np.sum(p_safe_all * np.log2(p_safe_all))
        total_entropies.append(h_all)
        
    n_values = np.arange(len(stack_history))
    
    plt.figure(figsize=(12, 6))
    plt.plot(n_values, top_entropies, 'g-', linewidth=2, label='Top-of-Stack Entropy')
    plt.title(f'Top-of-Stack Entropy over Full Sequence (L={prompt_len})', fontsize=16)
    plt.xlabel('Token Number (Timestep)')
    plt.ylabel('Entropy (bits)')
    plt.axvline(x=prompt_len, color='black', linestyle='--', label='Phase Switch (Delimiter)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(file_path_top)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(n_values, total_entropies, 'm-', linewidth=2, label='Total Stack Entropy')
    plt.title(f'Total Stack Entropy over Full Sequence (L={prompt_len})', fontsize=16)
    plt.xlabel('Token Number (Timestep)')
    plt.ylabel('Entropy (bits)')
    plt.axvline(x=prompt_len, color='black', linestyle='--', label='Phase Switch (Delimiter)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(file_path_total)
    plt.close()

def plot_comparative_entropy(stack_hist_1, stack_hist_2, label1, label2, prompt_len, file_path):
    """
    Plots the top-of-stack entropy for two different models on the same axes.
    """
    def get_top_entropy(stack_history):
        stack_history = np.array(stack_history).squeeze()
        decoding_start = prompt_len + 1
        if decoding_start >= len(stack_history):
            return []
        decoding_stacks = stack_history[decoding_start:]
        top_entropies = []
        for t in range(len(decoding_stacks)):
            p_top = decoding_stacks[t, 0, :]
            p_safe_top = p_top[p_top > 1e-9]
            h_top = -np.sum(p_safe_top * np.log2(p_safe_top))
            top_entropies.append(h_top)
        return top_entropies

    ent_1 = get_top_entropy(stack_hist_1)
    ent_2 = get_top_entropy(stack_hist_2)
    
    n_values_1 = np.arange(1, len(ent_1) + 1)
    n_values_2 = np.arange(1, len(ent_2) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_values_1, ent_1, 'b-', linewidth=2, label=label1)
    plt.plot(n_values_2, ent_2, 'r-', linewidth=2, label=label2)
    
    plt.title(f'Comparative Stack Entropy During Decoding (L={prompt_len})', fontsize=16)
    plt.xlabel('Decoding Step (Distance from phase switch)')
    plt.ylabel('Top-of-Stack Entropy (bits)')
    
    # We hardcode 60 here as the known training sequence length
    if prompt_len > 60:
        plt.axvline(x=60, color='red', linestyle='--', label='Training Horizon (L=60)')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(file_path)
    plt.close()

