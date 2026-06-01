import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Local imports
from minimal_models import MinimalStackRNN
from data_gen import generate_rev_trace

OUTPUT_DIR = "../results/reversal/phase_change_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_phase_change(steps=15000):
    model = MinimalStackRNN()
    key = jax.random.PRNGKey(42)
    params = model.init(key, jnp.zeros((1, 5), dtype=jnp.int32))['params']
    
    tx = optax.adam(1e-3)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    # Metrics to track
    history = {
        "loss": [],
        "grad_norm_Wq": [], # State Transition
        "grad_norm_Wa": [], # Memory Actions
        "grad_norm_Wb": [], # Buffer Output
        "weight_norm_Wq": [],
        "weight_norm_Wa": [],
        "weight_norm_Wb": [],
        "Wq_eq_trigger": []  # The specific weight connecting x_EQ to q_WRITE
    }

    @jax.jit
    def train_step(state, batch):
        inputs, targets, mask = batch
        def loss_fn(params):
            logits, _ = state.apply_fn({'params': params}, x=inputs)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            return (loss * mask).sum() / jnp.maximum(mask.sum(), 1e-9)
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        
        # Calculate norms for each head
        cell_grads = grads['ScanMinimalStackRNNCell_0']
        gn_q = jnp.linalg.norm(cell_grads['Wq']['kernel'])
        gn_a = jnp.linalg.norm(cell_grads['Wa']['kernel'])
        gn_b = jnp.linalg.norm(cell_grads['Wb']['kernel'])
        
        return state.apply_gradients(grads=grads), loss, (gn_q, gn_a, gn_b)

    print("Starting Phase Change Analysis...")
    for i in range(steps + 1):
        batch = generate_rev_trace(64, np.random.randint(10, 40))
        state, loss, g_norms = train_step(state, batch)
        
        # Record weights
        cell_w = state.params['ScanMinimalStackRNNCell_0']
        
        if i % 10 == 0:
            history["loss"].append(loss)
            history["grad_norm_Wq"].append(g_norms[0])
            history["grad_norm_Wa"].append(g_norms[1])
            history["grad_norm_Wb"].append(g_norms[2])
            history["weight_norm_Wq"].append(jnp.linalg.norm(cell_w['Wq']['kernel']))
            history["weight_norm_Wa"].append(jnp.linalg.norm(cell_w['Wa']['kernel']))
            history["weight_norm_Wb"].append(jnp.linalg.norm(cell_w['Wb']['kernel']))
            # Index 6 is x_EQ, index 1 is q_WRITE
            # Kernel is (In, Out)
            history["Wq_eq_trigger"].append(cell_w['Wq']['kernel'][6, 1])

        if i % 1000 == 0:
            print(f"Step {i} | Loss: {loss:.4f} | Wq_EQ: {history['Wq_eq_trigger'][-1]:.4f} | Grad_Wa: {g_norms[1]:.6f}")

    # --- Plotting ---
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    steps_ax = np.arange(len(history["loss"])) * 10
    
    # 1. Loss
    axes[0].plot(steps_ax, history["loss"], color='black', linewidth=2)
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("The Q=2 Phase Change")
    axes[0].grid(True, alpha=0.3)
    
    # 2. Gradient Norms (Log Scale)
    axes[1].plot(steps_ax, history["grad_norm_Wq"], label="Wq (State)", alpha=0.8)
    axes[1].plot(steps_ax, history["grad_norm_Wa"], label="Wa (Memory)", alpha=0.8)
    axes[1].plot(steps_ax, history["grad_norm_Wb"], label="Wb (Buffer)", alpha=0.8)
    axes[1].set_yscale('log')
    axes[1].set_ylabel("Gradient Norms")
    axes[1].legend()
    axes[1].grid(True, which="both", ls="-", alpha=0.2)
    
    # 3. Weight Norms
    axes[2].plot(steps_ax, history["weight_norm_Wq"], label="Wq (State)")
    axes[2].plot(steps_ax, history["weight_norm_Wa"], label="Wa (Memory)")
    axes[2].plot(steps_ax, history["weight_norm_Wb"], label="Wb (Buffer)")
    axes[2].set_ylabel("Weight Norms")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. State Trigger Signal
    axes[3].plot(steps_ax, history["Wq_eq_trigger"], color='red', label="x_EQ -> q_WRITE Weight")
    axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[3].set_ylabel("Weight Magnitude")
    axes[3].set_xlabel("Step")
    axes[3].set_title("Emergence of the State Transition Trigger")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "phase_change_diagnostics.png"))
    print(f"Analysis complete. Plot saved to {OUTPUT_DIR}/phase_change_diagnostics.png")

if __name__ == "__main__":
    analyze_phase_change()
