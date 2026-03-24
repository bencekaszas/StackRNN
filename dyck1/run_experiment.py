import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import linen as nn
import optax
import matplotlib.pyplot as plt
import numpy as np
import os

from constants import *
from data_gen import generate_dyck_batch, generate_fixed_batch, generate_dyck_string
from models import DyckStackRNN, DyckStackRNNCell
from visualize import (evaluate_and_visualize_dyck, plot_dyck_stack_viz, 
                            plot_dyck_state_trajectory, plot_dyck_ood_generalization)

OUTPUT_DIR = "/results/dyck1/8D_state_3"
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

def evaluate_dyck(state, prompt, max_len=100, hard_actions=True):
    """Auto-regressive decoding for Dyck completion."""
    batch_size = prompt.shape[0]
    
    # 1. Encoder Phase
    _, carry = state.apply_fn({'params': state.params}, x=prompt, hard_actions=hard_actions)
    
    # 2. Decoder Phase
    generated_sequence = []
    decoder_input = jnp.full((batch_size, 1), DYCK_EQ, dtype=jnp.int32)
    
    cell = DyckStackRNNCell(hard_actions=hard_actions)
    
    # Find the correct submodule key dynamically
    cell_key = [k for k in state.params.keys() if 'Cell' in k][0]
    cell_params = state.params[cell_key]

    for _ in range(max_len):
        x_emb = nn.Embed(DYCK_VOCAB_SIZE, HIDDEN_DIM, name="input_embed").apply({'params': state.params['input_embed']}, decoder_input)
        
        # Call cell with the extracted params
        carry, (logits, _) = cell.apply({'params': cell_params}, carry, x_emb[:, 0])
        
        next_token = jnp.argmax(logits, axis=-1)
        generated_sequence.append(next_token)
        
        if (next_token == DYCK_EOS).all():
            break
        decoder_input = next_token[:, None]

    return jnp.concatenate(generated_sequence, axis=0)

def run_ood_evaluation(state, test_lengths, hard_actions, n_samples=50):
    results = {}
    for L in test_lengths:
        correct_predictions = 0
        token_accuracies = []
        for _ in range(n_samples):
            prefix, expected_suffix = generate_dyck_string(STACK_DEPTH-10, L)
            expected_output = np.array(expected_suffix + [DYCK_EOS])
            
            generated = evaluate_dyck(state, jnp.array([prefix]), max_len=len(expected_output)+5, hard_actions=hard_actions)
            
            # Sequence Accuracy
            is_correct = False
            if len(generated) >= len(expected_output):
                if np.array_equal(generated[:len(expected_output)], expected_output):
                    is_correct = True
            if is_correct:
                correct_predictions += 1
            
            # Token Accuracy
            gen_trimmed = generated[:len(expected_output)]
            if len(gen_trimmed) < len(expected_output):
                gen_trimmed = np.concatenate([gen_trimmed, [DYCK_PAD] * (len(expected_output) - len(gen_trimmed))])
            token_acc = (gen_trimmed == expected_output).mean()
            token_accuracies.append(token_acc)
            
        seq_acc = correct_predictions / n_samples
        tok_acc = np.mean(token_accuracies)
        results[L] = (seq_acc, tok_acc)
        print(f"Len {L}: Seq Acc: {seq_acc:.2%} | Tok Acc: {tok_acc:.2%}")
    return results

if __name__ == "__main__":
    model = DyckStackRNN()
    
    print("\n=== Training Dyck-1 StackRNN ===")
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)
    state = create_train_state(model, key, LEARNING_RATE, dummy_input)

    losses = []
    accuracies = []

    for step in range(STEPS + 1):
        rand_len = np.random.randint(10, SEQ_LENGTH + 1)
        batch = generate_dyck_batch(BATCH_SIZE, rand_len)
        state, loss, acc = train_step(state, batch)
        losses.append(loss)
        accuracies.append(acc)
        
        if step % 500 == 0:
            print(f"Step {step} | Loss: {loss:.4f} | Acc: {acc:.2%}")

    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(OUTPUT_DIR, "training_loss_curve.png"))

    plt.figure(figsize=(10, 6))
    plt.plot(accuracies)
    plt.title("Training Accuracy Curve")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(OUTPUT_DIR, "training_accuracy_curve.png"))



    TEST_LENGTHS = [20, 40, 60, 100, 200, 300, 400, 500]
    
    print("\n--- OOD Evaluation (Hard Actions) ---")
    hard_results = run_ood_evaluation(state, TEST_LENGTHS, hard_actions=True)
    plot_dyck_ood_generalization(hard_results, "OOD Generalization (Hard Stack Inference)", 
                                 os.path.join(OUTPUT_DIR, "ood_gen_hard.png"), max_train_len=SEQ_LENGTH)

    print("\n--- OOD Evaluation (Soft Actions) ---")
    soft_results = run_ood_evaluation(state, TEST_LENGTHS, hard_actions=False)
    plot_dyck_ood_generalization(soft_results, "OOD Generalization (Soft Stack Inference)", 
                                 os.path.join(OUTPUT_DIR, "ood_gen_soft.png"), max_train_len=SEQ_LENGTH)

    # --- Visualizations ---
    print("\n--- Generating Visualizations ---")
    for L in [20, 100]:
        prefix, _ = generate_dyck_string(STACK_DEPTH-10, L)
        prompt = jnp.array([prefix])
        
        # Hard Actions Visualization
        seq, stack_hist, act_hist, state_hist = evaluate_and_visualize_dyck(state, prompt, max_len=L+10, hard_actions=True)
        suffix = "_long" if L > 60 else ""
        plot_dyck_stack_viz(seq, stack_hist, act_hist, os.path.join(OUTPUT_DIR, f"stack_viz_hard{suffix}.png"))
        plot_dyck_state_trajectory(state_hist, L, os.path.join(OUTPUT_DIR, f"state_traj_hard{suffix}.png"))
        
        # Soft Actions Visualization
        seq, stack_hist, act_hist, state_hist = evaluate_and_visualize_dyck(state, prompt, max_len=L+10, hard_actions=False)
        plot_dyck_stack_viz(seq, stack_hist, act_hist, os.path.join(OUTPUT_DIR, f"stack_viz_soft{suffix}.png"))
        plot_dyck_state_trajectory(state_hist, L, os.path.join(OUTPUT_DIR, f"state_traj_soft{suffix}.png"))

    print(f"Results saved to {OUTPUT_DIR}")
