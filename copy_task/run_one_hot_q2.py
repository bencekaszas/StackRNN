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

from constants import *
from data_gen import generate_rev_trace, generate_fixed_batch
from models import StackRNN

# Import constants for plotting
from constants import ACT_PUSH_0, ACT_PUSH_1, ACT_POP, STACK_NULL
from visualise import (evaluate_and_visualize, plot_deepmind_style, 
                       plot_state_trajectory, plot_final_stack_distribution, 
                       plot_read_fidelity, plot_epsilon_analysis,
                       plot_fidelity_vs_theory)

OUTPUT_DIR = "../results/reversal/one_hot_Q2_softmax"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_train_state(model, key, learning_rate, dummy_input):
    params = model.init(key, dummy_input)['params']
    tx = optax.chain(optax.adam(learning_rate))
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def masked_loss(logits, targets, mask):
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return (loss * mask).sum() / jnp.maximum(mask.sum(), 1e-9)

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
    _, carry = state.apply_fn({'params': state.params}, x=prompt, hard_actions=hard_actions)
    decoder_input = jnp.full((prompt.shape[0], 1), VOCAB_EQ, dtype=jnp.int32)
    generated_sequence = []
    cell = StackRNN.cell_cls(hard_actions=hard_actions)
    input_proj_params = state.params.get('input_proj', None)

    for _ in range(max_len):
        x_oh = jax.nn.one_hot(decoder_input, VOCAB_SIZE)
        decoder_emb = nn.Dense(HIDDEN_DIM, name="input_proj").apply({'params': input_proj_params}, x_oh)
        carry, (logits, _) = cell.apply({'params': state.params['ScanStackRNNCell_0']}, carry, decoder_emb[:, 0])
        next_token = jnp.argmax(logits, axis=-1)
        generated_sequence.append(next_token)
        if (next_token == VOCAB_EOS).all(): break
        decoder_input = next_token[:, None]
    return jnp.concatenate(generated_sequence, axis=0)

if __name__ == "__main__":
    # SET UP: One-hot, Q=2
    # Ensure constants.py is Q=2
    model = StackRNN(use_one_hot_emb=True)
    
    print(f"\n=== Training One-Hot StackRNN (Q={NUM_STATES}) ===")
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((1, 2 * SEQ_LENGTH + 2), dtype=jnp.int32)
    state = create_train_state(model, key, LEARNING_RATE, dummy_input)

    losses, accs = [], []
    for step in range(STEPS + 1):
        rand_len = np.random.randint(10, SEQ_LENGTH + 1)
        batch = generate_rev_trace(BATCH_SIZE, rand_len)
        state, loss, acc = train_step(state, batch)
        losses.append(loss)
        accs.append(acc)
        if step % 500 == 0:
            print(f"Step {step} | Loss: {loss:.4f} | Acc: {acc:.2%}")

    # 1. Training Curves
    plt.figure(figsize=(10, 6))
    plt.plot(losses); plt.title("Training Loss Curve"); plt.xlabel("Step"); plt.ylabel("Loss")
    plt.savefig(os.path.join(OUTPUT_DIR, "training_loss_curve.pdf")); plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(accs); plt.title("Training Accuracy Curve"); plt.xlabel("Step"); plt.ylabel("Accuracy")
    plt.savefig(os.path.join(OUTPUT_DIR, "training_accuracy_curve.pdf")); plt.close()

    # 2. Visualizations
    for L in [40, 500]:
        print(f"--- Visualizing L={L} ---")
        prompt = generate_fixed_batch(1, L)
        full_seq, stack_hist, action_hist, state_hist = evaluate_and_visualize(state, prompt, max_len=L+10, hard_actions=False)
        
        suffix = "_long" if L > SEQ_LENGTH else ""
        plot_deepmind_style(full_seq, stack_hist, action_hist, os.path.join(OUTPUT_DIR, f"stack_visualization{suffix}.pdf"))
        plot_state_trajectory(state_hist, L, os.path.join(OUTPUT_DIR, f"state_trajectory{suffix}.pdf"))
        plot_read_fidelity(stack_hist, full_seq, L, os.path.join(OUTPUT_DIR, f"read_fidelity{suffix}.pdf"))
        
        # Epsilon Analysis
        plot_epsilon_analysis(full_seq, action_hist, L, 
                               os.path.join(OUTPUT_DIR, f"epsilon_dist{suffix}.pdf"),
                               os.path.join(OUTPUT_DIR, f"epsilon_time{suffix}.pdf"),
                               os.path.join(OUTPUT_DIR, f"epsilons{suffix}.npy"))

    print(f"Results saved to {OUTPUT_DIR}")
