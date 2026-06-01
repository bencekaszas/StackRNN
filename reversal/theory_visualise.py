import jax
import jax.numpy as jnp
from flax.core import unfreeze, freeze
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from minimal_models import MinimalStackRNN, MinimalStackRNNCell
from data_gen import generate_fixed_batch, generate_rev_trace
from constants import VOCAB_EOS

# Basis: (r_NULL, r_0, r_1, x_PAD, x_0, x_1, x_EQ, x_EOS, q_READ, q_WRITE)
BASIS_LABELS = ["r_NULL", "r_0", "r_1", "x_PAD", "x_0", "x_1", "x_EQ", "x_EOS", "q_READ", "q_WRITE"]

OUTPUT_DIR = "../results/reversal/theoretical_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_lemma_weights(omega=100.0):
    Wq = jnp.zeros((2, 10))
    Wq = Wq.at[0, 6].set(-2.0)
    Wq = Wq.at[0, 8].set(1.0)
    Wq = Wq.at[1, 6].set(2.0)
    Wq = Wq.at[1, 9].set(1.0)
    Wa = jnp.zeros((4, 10))
    Wa = Wa.at[0, 0].set(1.0)
    Wa = Wa.at[0, 9].set(2.0)
    Wa = Wa.at[1, 4].set(2.0)
    Wa = Wa.at[1, 8].set(2.0)
    Wa = Wa.at[2, 5].set(2.0)
    Wa = Wa.at[2, 8].set(2.0)
    Wa = Wa.at[3, 1].set(1.0)
    Wa = Wa.at[3, 2].set(1.0)
    Wa = Wa.at[3, 6].set(4.0)
    Wa = Wa.at[3, 9].set(2.0)
    Wb = jnp.zeros((5, 10))
    Wb = Wb.at[0, 8].set(3.0)
    Wb = Wb.at[0, 6].set(-4.0)
    Wb = Wb.at[1, 1].set(2.0)
    Wb = Wb.at[1, 6].set(1.0)
    Wb = Wb.at[1, 9].set(1.0)
    Wb = Wb.at[2, 2].set(2.0)
    Wb = Wb.at[2, 6].set(1.0)
    Wb = Wb.at[2, 9].set(1.0)
    Wb = Wb.at[4, 0].set(2.0)
    Wb = Wb.at[4, 9].set(1.0)
    return Wq * omega, Wa * omega, Wb * omega

def run_eval_and_collect(params, length=40):
    prompt_bits = np.random.randint(1, 3, size=length).tolist()
    expected = prompt_bits[::-1] + [4]
    stack = jnp.zeros((1, 600, 3)).at[:, :, 0].set(1.0)
    state = jnp.zeros((1, 2)).at[:, 0].set(1.0)
    carry = (stack, state)
    cell = MinimalStackRNNCell(hard_actions=False)
    cell_params = params['ScanMinimalStackRNNCell_0']
    
    full_seq = []
    action_hist = []
    stack_hist = []
    state_hist = []
    
    # Encode
    for b in prompt_bits:
        full_seq.append(b)
        stack_hist.append(carry[0])
        state_hist.append(carry[1])
        x_oh = jax.nn.one_hot(jnp.array([b]), 5)
        carry, (_, acts) = cell.apply({'params': cell_params}, carry, x_oh)
        action_hist.append(acts)
    
    # Decode
    decoder_input = 3
    for _ in range(length + 5):
        full_seq.append(decoder_input)
        stack_hist.append(carry[0])
        state_hist.append(carry[1])
        x_oh = jax.nn.one_hot(jnp.array([decoder_input]), 5)
        carry, (logits, acts) = cell.apply({'params': cell_params}, carry, x_oh)
        action_hist.append(acts)
        next_tok = jnp.argmax(logits, axis=-1).item()
        decoder_input = next_tok
        if next_tok == 4:
            full_seq.append(4)
            stack_hist.append(carry[0])
            state_hist.append(carry[1])
            break
            
    return np.array(full_seq), np.array(stack_hist), np.array(action_hist), np.array(state_hist).squeeze()

def plot_theory_viz(full_seq, stack_hist, action_hist, state_hist, prefix):
    # Reuse plotting logic from visualize.py
    from visualise import plot_deepmind_style, plot_state_trajectory
    plot_deepmind_style(full_seq, stack_hist, action_hist, os.path.join(OUTPUT_DIR, f"{prefix}_stack.png"))
    plot_state_trajectory(state_hist, len(full_seq)//2, os.path.join(OUTPUT_DIR, f"{prefix}_state.png"))

if __name__ == "__main__":
    model = MinimalStackRNN()
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.zeros((1, 5), dtype=jnp.int32))['params']
    Wq_v, Wa_v, Wb_v = get_lemma_weights(omega=100.0)
    new_params = unfreeze(params)
    new_params['ScanMinimalStackRNNCell_0']['Wq']['kernel'] = Wq_v.T
    new_params['ScanMinimalStackRNNCell_0']['Wa']['kernel'] = Wa_v.T
    new_params['ScanMinimalStackRNNCell_0']['Wb']['kernel'] = Wb_v.T
    new_params = freeze(new_params)
    
    # 1. Sequence Visualizations
    for L in [40, 500]:
        print(f"Generating visuals for L={L}...")
        seq, sh, ah, sth = run_eval_and_collect(new_params, length=L)
        suffix = "long" if L > 60 else "moderate"
        plot_theory_viz(seq, sh, ah, sth, suffix)
        
    # 2. OOD Accuracy Plot
    print("Generating OOD Accuracy Plot...")
    results = {}
    TEST_LENGTHS = [10, 20, 40, 60, 100, 200, 300, 400, 500]
    for L in TEST_LENGTHS:
        correct = 0
        from inspect_minimal import test_single_reversal
        for _ in range(20):
            correct += test_single_reversal(new_params, length=L)
        results[L] = correct / 20
    
    print("OOD Accuracy Results:")
    for L, acc in results.items():
        print(f"Length {L}: Accuracy {acc:.2%}")
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    plt.plot(list(results.keys()), list(results.values()), marker='o', label="Sequence Accuracy")
    plt.axvline(x=60, color='gray', linestyle='--', label="Max Train Length (60)")
    plt.title("OOD Generalization (Theoretical Lemma Construction)")
    plt.xlabel("Length")
    plt.ylabel("Accuracy")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "ood_generalization_plot.png"))
    print(f"All theoretical plots saved to {OUTPUT_DIR}")
