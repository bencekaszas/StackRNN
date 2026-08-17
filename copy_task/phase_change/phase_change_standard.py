import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Standard StackRNN with forced Q=2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stack_utils import soft_update_stack

# Local imports
from constants import *
from data_gen import generate_rev_trace

OUTPUT_DIR = "../results/reversal/phase_change_standard"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class Q2StackRNNCell(nn.Module):
    hard_actions: bool = False
    
    @nn.compact
    def __call__(self, carry, x_emb):
        stack, state_prev = carry
        stack_top = stack[:, 0]
        
        state_emb = nn.Dense(HIDDEN_DIM, name="state_embed")(state_prev)
        stack_top_emb = nn.Dense(HIDDEN_DIM, name="stack_top_embed")(stack_top)         
        flat_input = jnp.concatenate([x_emb, state_emb, stack_top_emb], axis=-1)
        
        logits_mem = nn.Dense(NUM_MEM_ACTIONS)(flat_input)
        logits_buf = nn.Dense(VOCAB_SIZE)(flat_input)
        logits_state = nn.Dense(2)(flat_input) # FORCED Q=2
        
        action_probs = nn.softmax(logits_mem)
        if self.hard_actions:
            max_act = jnp.argmax(action_probs, axis=-1)
            action_probs = jax.nn.one_hot(max_act, NUM_MEM_ACTIONS)
        
        stack_new, _ = jax.vmap(soft_update_stack)(stack, action_probs)
        next_state = jnp.tanh(logits_state)
        
        new_carry = (stack_new, next_state)
        return new_carry, (logits_buf, action_probs)

class Q2StackRNN(nn.Module):
    @nn.compact
    def __call__(self, x, hard_actions=False):
        batch_size, seq_len = x.shape
        x_emb = nn.Embed(VOCAB_SIZE, HIDDEN_DIM, name="input_embed")(x)
        
        init_stack = jnp.zeros((batch_size, STACK_DEPTH, STACK_VOCAB_SIZE))
        init_stack = init_stack.at[:, :, 0].set(1.0)
        init_state = jnp.zeros((batch_size, 2), dtype=jnp.float32)
        carry = (init_stack, init_state)
        
        scan_layer = nn.scan(Q2StackRNNCell, variable_broadcast="params", 
                             split_rngs={"params": False}, in_axes=1, out_axes=1)
        
        final_carry, (logits_buf, action_probs) = scan_layer(hard_actions=hard_actions)(carry, x_emb)
        return logits_buf, action_probs

def analyze():
    model = Q2StackRNN()
    key = jax.random.PRNGKey(42)
    params = model.init(key, jnp.zeros((1, 10), dtype=jnp.int32))['params']
    
    tx = optax.adam(LEARNING_RATE)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    history = {
        "loss": [],
        "grad_norm_embed": [],
        "grad_norm_mem": [],
        "grad_norm_state": [],
        "entropy": [],
        "eq_emb_norm": []
    }

    @jax.jit
    def train_step(state, batch):
        inputs, targets, mask = batch
        def loss_fn(params):
            logits, action_probs = state.apply_fn({'params': params}, x=inputs)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            loss = (loss * mask).sum() / jnp.maximum(mask.sum(), 1e-9)
            entropy = -jnp.sum(action_probs * jnp.log(action_probs + 1e-9), axis=-1).mean()
            return loss, (loss, entropy)
        
        (loss, (loss_val, entropy)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        
        # Norms
        cell_grads = grads['ScanQ2StackRNNCell_0']
        gn_embed = jnp.linalg.norm(grads['input_embed']['embedding'])
        gn_mem = jnp.linalg.norm(cell_grads['Dense_0']['kernel'])
        gn_state = jnp.linalg.norm(cell_grads['Dense_2']['kernel'])
        
        return state.apply_gradients(grads=grads), loss_val, entropy, (gn_embed, gn_mem, gn_state)

    print("Starting Standard Q=2 Analysis (Learned Embeddings)...")
    STEPS_ANALYSIS = 15000
    for i in range(STEPS_ANALYSIS + 1):
        batch = generate_rev_trace(BATCH_SIZE, np.random.randint(10, 40))
        state, loss, entropy, g_norms = train_step(state, batch)
        
        if i % 20 == 0:
            history["loss"].append(loss)
            history["entropy"].append(entropy)
            history["grad_norm_embed"].append(g_norms[0])
            history["grad_norm_mem"].append(g_norms[1])
            history["grad_norm_state"].append(g_norms[2])
            # Eq token is index 3
            history["eq_emb_norm"].append(jnp.linalg.norm(state.params['input_embed']['embedding'][3]))

        if i % 1000 == 0:
            print(f"Step {i} | Loss: {loss:.4f} | Entropy: {entropy:.4f}")

    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    steps_ax = np.arange(len(history["loss"])) * 20
    
    axes[0].plot(steps_ax, history["loss"], color='black')
    axes[0].set_title("Q=2 Phase Change (Standard Setting)")
    axes[0].set_ylabel("CE Loss")
    
    axes[1].plot(steps_ax, history["entropy"], color='blue')
    axes[1].set_title("Action Entropy (Sharpness of Memory Policy)")
    axes[1].set_ylabel("Entropy")
    
    axes[2].plot(steps_ax, history["grad_norm_embed"], label="Embed")
    axes[2].plot(steps_ax, history["grad_norm_mem"], label="Memory")
    axes[2].plot(steps_ax, history["grad_norm_state"], label="State")
    axes[2].set_yscale('log')
    axes[2].set_title("Gradient Norms")
    axes[2].legend()
    
    axes[3].plot(steps_ax, history["eq_emb_norm"], color='red')
    axes[3].set_title("Norm of '=' Token Embedding")
    axes[3].set_xlabel("Step")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "phase_change_standard.pdf"))
    print(f"Analysis complete. Plot saved to {OUTPUT_DIR}/phase_change_standard.pdf")

if __name__ == "__main__":
    analyze()
