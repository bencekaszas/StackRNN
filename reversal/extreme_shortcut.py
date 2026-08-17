import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
from models import StackRNN
from data_gen import generate_rev_trace
from constants import *
from run_experiment import create_train_state, train_step
from visualise import evaluate_and_visualize, plot_stack_entropy, plot_sequence_entropy
import matplotlib.pyplot as plt

SHORTCUT_L = 8
OUTPUT_DIR = "../results/reversal/extreme_shortcut"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_extreme_shortcut():
    key = jax.random.PRNGKey(123)
    dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)
    
    # Initialize Q=64 model
    model = StackRNN(num_states=64)
    state = create_train_state(model, key, LEARNING_RATE, dummy_input)
    
    print(f"=== Training Extreme Shortcut Model (Q=64, L_max={SHORTCUT_L}) ===")
    for step in range(1, 10001):
        batch = generate_rev_trace(BATCH_SIZE, SHORTCUT_L)
        state, loss, acc = train_step(state, batch)
        if step % 500 == 0:
            print(f"Step {step} | Loss: {loss:.4f} | Acc: {acc*100:.2f}%")
        if loss < 0.005 and acc > 0.99:
            print(f"Converged at step {step}!")
            break

    print("\n--- Evaluating OOD ---")
    lengths = [4, 8, 12, 16, 20, 30, 40]
    
    seq_accs = []
    tok_accs = []
    for L in lengths:
        correct = 0
        token_acc = []
        for _ in range(50):
            bits = np.random.randint(1, 3, size=L)
            prompt = np.concatenate([bits, [VOCAB_EQ]])
            expected = np.concatenate([bits[::-1], [VOCAB_EOS]])
            
            generated, stack_hist, action_hist, *_ = evaluate_and_visualize(state, jnp.array([prompt]), max_len=L+5, hard_actions=False, num_states=64)
            gen_suffix = generated[len(prompt):len(prompt)+len(expected)]
            
            if np.array_equal(gen_suffix, expected):
                correct += 1
            matches = (gen_suffix == expected[:len(gen_suffix)]).sum()
            token_acc.append(matches / len(expected))
            
            # Save entropy for L=40
            if L == 40 and correct == 0 and len(token_acc) == 1:
                plot_sequence_entropy(stack_hist, len(prompt), f"{OUTPUT_DIR}/shortcut_entropy_top.pdf", f"{OUTPUT_DIR}/shortcut_entropy_total.pdf")
                
        seq_accs.append(correct / 50.0)
        tok_accs.append(np.mean(token_acc))
        print(f"Len {L}: Seq Acc: {seq_accs[-1]*100:.2f}% | Tok Acc: {tok_accs[-1]*100:.2f}%")
        
    # Plot OOD curve
    plt.figure(figsize=(8,5))
    plt.plot(lengths, seq_accs, marker='o', label='Sequence Acc')
    plt.plot(lengths, tok_accs, marker='x', label='Token Acc')
    plt.axvline(x=SHORTCUT_L, color='r', linestyle='--', label=f'Max Train Length ({SHORTCUT_L})')
    plt.title("OOD Generalization (Extreme Shortcut)")
    plt.xlabel("Sequence Length")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/extreme_shortcut_ood.pdf")
    plt.close()
    print("Done!")

if __name__ == "__main__":
    run_extreme_shortcut()
