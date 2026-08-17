import jax
import jax.numpy as jnp
from flax.training import train_state
from flax.core import unfreeze, freeze
import optax
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

# Basis: (r_NULL, r_0, r_1, x_PAD, x_0, x_1, x_EQ, x_EOS, q_READ, q_WRITE)
BASIS_LABELS = [
    "r_NULL", "r_0", "r_1", 
    "x_PAD", "x_0", "x_1", "x_EQ", "x_EOS",
    "q_READ", "q_WRITE"
]

# Local imports
from minimal_models import MinimalStackRNN, MinimalStackRNNCell
from data_gen import generate_rev_trace

OUTPUT_DIR = "../results/reversal/lemma_verification"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_lemma_weights(omega=50.0):
    Wq = jnp.zeros((2, 10))
    Wq = Wq.at[0, 6].set(-2.0)
    Wq = Wq.at[0, 8].set(1.0)
    Wq = Wq.at[1, 6].set(2.0)
    Wq = Wq.at[1, 9].set(1.0)
    
    Wa = jnp.zeros((4, 10))
    Wa = Wa.at[0, 0].set(1.0) # NOOP: r_NULL
    Wa = Wa.at[0, 9].set(2.0) # NOOP: q_WRITE
    Wa = Wa.at[1, 4].set(2.0) # PUSH0: x_0
    Wa = Wa.at[1, 8].set(2.0) # PUSH0: q_READ
    Wa = Wa.at[2, 5].set(2.0) # PUSH1: x_1
    Wa = Wa.at[2, 8].set(2.0) # PUSH1: q_READ
    Wa = Wa.at[3, 1].set(1.0) # POP: r_0
    Wa = Wa.at[3, 2].set(1.0) # POP: r_1
    Wa = Wa.at[3, 6].set(4.0) # POP: x_EQ
    Wa = Wa.at[3, 9].set(2.0) # POP: q_WRITE
    
    Wb = jnp.zeros((5, 10))
    Wb = Wb.at[0, 8].set(3.0) # PAD: q_READ
    Wb = Wb.at[0, 6].set(-4.0) # INHIBIT PAD ON EQ
    Wb = Wb.at[1, 1].set(2.0) # 0: r_0
    Wb = Wb.at[1, 6].set(1.0) # IMMEDIATE 0 ON EQ
    Wb = Wb.at[1, 9].set(1.0) # 0: q_WRITE
    Wb = Wb.at[2, 2].set(2.0) # 1: r_1
    Wb = Wb.at[2, 6].set(1.0) # IMMEDIATE 1 ON EQ
    Wb = Wb.at[2, 9].set(1.0) # 1: q_WRITE
    Wb = Wb.at[4, 0].set(2.0) # EOS: r_NULL
    Wb = Wb.at[4, 9].set(1.0) # EOS: q_WRITE
    
    return Wq * omega, Wa * omega, Wb * omega

def plot_weight_matrices(params, title, filename):
    cell_params = params['ScanMinimalStackRNNCell_0']
    Wq = cell_params['Wq']['kernel'].T
    Wa = cell_params['Wa']['kernel'].T
    Wb = cell_params['Wb']['kernel'].T
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    sns.heatmap(Wq, ax=axes[0], annot=False, xticklabels=BASIS_LABELS, yticklabels=["READ", "WRITE"], cmap="RdBu_r", center=0)
    axes[0].set_title(f"{title}: State Transition (Wq)")
    sns.heatmap(Wa, ax=axes[1], annot=False, xticklabels=BASIS_LABELS, yticklabels=["NOOP", "PUSH0", "PUSH1", "POP"], cmap="RdBu_r", center=0)
    axes[1].set_title(f"{title}: Memory Action (Wa)")
    sns.heatmap(Wb, ax=axes[2], annot=False, xticklabels=BASIS_LABELS, yticklabels=["PAD", "0", "1", "EQ", "EOS"], cmap="RdBu_r", center=0)
    axes[2].set_title(f"{title}: Buffer Output (Wb)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def test_single_reversal(params, length=2):
    prompt_bits = np.random.randint(1, 3, size=length).tolist()
    expected = prompt_bits[::-1] + [4]
    
    stack = jnp.zeros((1, 600, 3))
    stack = stack.at[:, :, 0].set(1.0)
    state = jnp.zeros((1, 2))
    state = state.at[:, 0].set(1.0)
    carry = (stack, state)
    
    cell = MinimalStackRNNCell(hard_actions=True)
    cell_params = params['ScanMinimalStackRNNCell_0']
    
    for b in prompt_bits:
        x_oh = jax.nn.one_hot(jnp.array([b]), 5)
        carry, _ = cell.apply({'params': cell_params}, carry, x_oh)
    
    decoder_input = 3
    generated = []
    for _ in range(length + 5):
        x_oh = jax.nn.one_hot(jnp.array([decoder_input]), 5)
        carry, (logits, _) = cell.apply({'params': cell_params}, carry, x_oh)
        next_tok = jnp.argmax(logits, axis=-1).item()
        generated.append(next_tok)
        decoder_input = next_tok
        if next_tok == 4: break
    return generated == expected

def train_minimal(steps=10000, l1_lambda=1e-4):
    model = MinimalStackRNN()
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 5), dtype=jnp.int32)
    params = model.init(key, dummy_input)['params']
    tx = optax.adam(1e-3)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    history = {"loss": [], "gn_q": [], "gn_a": [], "gn_b": []}
    
    @jax.jit
    def train_step(state, batch):
        inputs, targets, mask = batch
        def loss_fn(params):
            logits, _ = state.apply_fn({'params': params}, x=inputs)
            ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            ce_loss = (ce_loss * mask).sum() / jnp.maximum(mask.sum(), 1e-9)
            l1_loss = sum(jnp.abs(p).sum() for p in jax.tree_util.tree_leaves(params))
            return ce_loss + l1_lambda * l1_loss, ce_loss
            
        (total_loss, ce_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        
        # Track Grads
        cell_grads = grads['ScanMinimalStackRNNCell_0']
        gn_q = jnp.linalg.norm(cell_grads['Wq']['kernel'])
        gn_a = jnp.linalg.norm(cell_grads['Wa']['kernel'])
        gn_b = jnp.linalg.norm(cell_grads['Wb']['kernel'])
        
        return state.apply_gradients(grads=grads), ce_loss, (gn_q, gn_a, gn_b)

    print(f"Training Minimal Model with L1 Regularization (lambda={l1_lambda})...")
    for i in range(steps + 1):
        batch = generate_rev_trace(64, np.random.randint(10, 40))
        state, ce_loss, g_norms = train_step(state, batch)
        
        if i % 100 == 0:
            history["loss"].append(ce_loss)
            history["gn_q"].append(g_norms[0])
            history["gn_a"].append(g_norms[1])
            history["gn_b"].append(g_norms[2])

        if i % 1000 == 0:
            print(f"Step {i} | CE Loss: {ce_loss:.4f} | GN_Wa: {g_norms[1]:.6f}")
            
    # Plot Phase Change Diagnostics
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(history["loss"], color='black')
    plt.title("Phase Change: Loss")
    plt.ylabel("CE Loss")
    plt.subplot(2, 1, 2)
    plt.plot(history["gn_q"], label="Grad Wq (State)")
    plt.plot(history["gn_a"], label="Grad Wa (Memory)")
    plt.plot(history["gn_b"], label="Grad Wb (Buffer)")
    plt.yscale('log')
    plt.title("Phase Change: Gradient Norms")
    plt.ylabel("Norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "phase_change_diagnostics.pdf"))
    
    return state.params

def analyze_correlation(theoretical, learned):
    def get_flat(p):
        cell = p['ScanMinimalStackRNNCell_0']
        return jnp.concatenate([
            cell['Wq']['kernel'].flatten(),
            cell['Wa']['kernel'].flatten(),
            cell['Wb']['kernel'].flatten()
        ])
    v_theo = get_flat(theoretical)
    v_learn = get_flat(learned)
    v_theo = v_theo / jnp.linalg.norm(v_theo)
    v_learn = v_learn / jnp.linalg.norm(v_learn)
    return jnp.dot(v_theo, v_learn)

if __name__ == "__main__":
    model = MinimalStackRNN()
    key = jax.random.PRNGKey(42)
    init_params = model.init(key, jnp.zeros((1, 5), dtype=jnp.int32))['params']
    
    OMEGA = 50.0
    Wq_val, Wa_val, Wb_val = get_lemma_weights(omega=OMEGA)
    theo_params = unfreeze(init_params)
    theo_params['ScanMinimalStackRNNCell_0']['Wq']['kernel'] = Wq_val.T
    theo_params['ScanMinimalStackRNNCell_0']['Wa']['kernel'] = Wa_val.T
    theo_params['ScanMinimalStackRNNCell_0']['Wb']['kernel'] = Wb_val.T
    theo_params = freeze(theo_params)
    
    plot_weight_matrices(theo_params, f"Theoretical Construction (omega={OMEGA})", "theoretical_weights.pdf")
    
    print("--- 1. Theoretical Construction Accuracy ---")
    for L in [5, 100, 500]:
        acc = sum(test_single_reversal(theo_params, length=L) for _ in range(20)) / 20
        print(f"Length {L}: Acc={acc:.2%}")
    
    print("\n--- 2. SGD Learning ---")
    learned_params = train_minimal()
    plot_weight_matrices(learned_params, "SGD Learned Solution", "learned_weights.pdf")
    
    print("\n--- 3. Learned Solution Accuracy ---")
    for L in [5, 100, 500]:
        acc = sum(test_single_reversal(learned_params, length=L) for _ in range(20)) / 20
        print(f"Length {L}: Acc={acc:.2%}")
        
    similarity = analyze_correlation(theo_params, learned_params)
    print(f"\nWeight Correlation (Cosine Similarity): {similarity:.4f}")
    print(f"Plots saved to {OUTPUT_DIR}")
