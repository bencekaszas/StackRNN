import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import sys

from constants import *
from data_gen import generate_rev_trace, generate_fixed_batch
from models import StackRNN
from visualise import evaluate_and_visualize, plot_comparative_entropy

OUTPUT_DIR = "../results/reversal/shortcut_experiment"
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

def train_model(model_name, num_states):
    print(f"\n=== Training StackRNN ({model_name}, Q={num_states}) ===")
    model = StackRNN(use_one_hot_emb=False, num_states=num_states)
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((1, 2 * SEQ_LENGTH + 2), dtype=jnp.int32)
    state = create_train_state(model, key, LEARNING_RATE, dummy_input)

    consecutive_perfect = 0
    MAX_STEPS = 20000 
    
    losses = []
    accs = []
    
    for step in range(MAX_STEPS + 1):
        rand_max_len = np.random.randint(10, SEQ_LENGTH + 1)
        batch = generate_rev_trace(BATCH_SIZE, rand_max_len)
        state, loss, acc = train_step(state, batch)
        
        losses.append(loss)
        accs.append(acc)
        
        if acc >= 0.999:
            consecutive_perfect += 1
        else:
            consecutive_perfect = 0
            
        if step % 500 == 0:
            print(f"[{model_name}] Step {step} | Train Loss: {loss:.4f} | Train Acc: {acc:.2%}")
            
        if consecutive_perfect >= 100 and step > 4000:
            print(f"[{model_name}] Reached stable perfect training accuracy at step {step}!")
            break

    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(f"Training Loss Curve ({model_name})")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(OUTPUT_DIR, f"training_loss_Q{num_states}.pdf"))
    plt.close()

    return state, model

def evaluate_custom(state, prompt, max_len=100, hard_actions=False, num_states=2):
    _, carry = state.apply_fn({'params': state.params}, x=prompt, hard_actions=hard_actions)
    decoder_input = jnp.full((prompt.shape[0], 1), VOCAB_EQ, dtype=jnp.int32)
    generated_sequence = []
    cell = StackRNN.cell_cls(hard_actions=hard_actions, num_states=num_states)
    embed_params = state.params.get('input_embed', None)
    input_proj_params = state.params.get('input_proj', None)

    for _ in range(max_len):
        if embed_params is not None:
            import flax.linen as nn
            decoder_emb = nn.Embed(VOCAB_SIZE, HIDDEN_DIM, name="input_embed").apply({'params': embed_params}, decoder_input)
        else:
            import flax.linen as nn
            decoder_emb = jax.nn.one_hot(decoder_input, VOCAB_SIZE)
            decoder_emb = nn.Dense(HIDDEN_DIM, name="input_proj").apply({'params': input_proj_params}, decoder_emb)
        
        carry, (logits, action_probs) = cell.apply({'params': state.params['ScanStackRNNCell_0']}, carry, decoder_emb[:, 0])
        next_token = jnp.argmax(logits, axis=-1)
        generated_sequence.append(next_token)
        if (next_token == VOCAB_EOS).all():
            break
        decoder_input = next_token[:, None]

    return jnp.concatenate(generated_sequence, axis=0)

if __name__ == "__main__":
    # 1. Train models
    state_q64, model_q64 = train_model("Shortcut Model", 64)
    state_q2, model_q2 = train_model("Algorithmic Model", 2)
    
    # 2. Evaluate on OOD sequence L=500
    print(f"\n--- Evaluating OOD and Generating Visualizations ---")
    VIS_L_LONG = 500
    vis_prompt_long = generate_fixed_batch(1, VIS_L_LONG)
    
    print("Evaluating Q=64...")
    _, stack_hist_q64, _, _, _ = evaluate_and_visualize(state_q64, vis_prompt_long, max_len=VIS_L_LONG+10, hard_actions=False, num_states=64)
    
    print("Evaluating Q=2...")
    _, stack_hist_q2, _, _, _ = evaluate_and_visualize(state_q2, vis_prompt_long, max_len=VIS_L_LONG+10, hard_actions=False, num_states=2)
    
    # 3. Comparative Entropy Plot
    print("Generating comparative entropy plot...")
    plot_comparative_entropy(
        stack_hist_q64, stack_hist_q2, 
        "Shortcut Model (Q=64)", "Algorithmic Model (Q=2)", 
        VIS_L_LONG, 
        os.path.join(OUTPUT_DIR, "comparative_entropy_L500.pdf")
    )
    
    # 4. Comparative OOD Accuracy
    print("Generating comparative OOD Accuracy plots...")
    TEST_LENGTHS = [10, 20, 40, 60, 100, 200, 300, 400, 500]
    
    def eval_model_ood(state, num_states):
        results = {}
        for L in TEST_LENGTHS:
            N_SAMPLES = 100 if L <= 100 else 50
            prompts = generate_fixed_batch(N_SAMPLES, L)
            correct_predictions = 0
            token_accuracies_l = []
            for i in range(N_SAMPLES):
                prompt = prompts[i:i+1, :]
                prompt_bits = prompt[0, :L]
                generated = evaluate_custom(state, jnp.array(prompt_bits[None, :]), max_len=L+10, hard_actions=False, num_states=num_states)
                
                generated_output = generated
                ground_truth = np.asarray(prompt_bits[::-1])
                ground_truth = np.concatenate([ground_truth, [VOCAB_EOS]])
                
                is_correct = False
                if len(generated_output) >= len(ground_truth):
                    if np.array_equal(generated_output[:len(ground_truth)], ground_truth):
                        if len(generated_output) == len(ground_truth) or generated_output[len(ground_truth)] == VOCAB_EOS:
                            is_correct = True
                if is_correct:
                    correct_predictions += 1
                
                if len(generated_output) > len(ground_truth):
                    generated_output = generated_output[:len(ground_truth)]
                elif len(generated_output) < len(ground_truth):
                    generated_output = np.concatenate([generated_output, [VOCAB_EOS] * (len(ground_truth) - len(generated_output))])
                token_accuracy = (generated_output == ground_truth).mean()
                token_accuracies_l.append(token_accuracy)
                    
            seq_acc = correct_predictions / N_SAMPLES
            token_acc = np.mean(token_accuracies_l)
            print(f"  Len {L}: Seq Acc: {seq_acc:.2%} | Token Acc: {token_acc:.2%}")
            results[L] = (seq_acc, token_acc)
        return results

    print("Evaluating Q=64 OOD Accuracy:")
    res_q64 = eval_model_ood(state_q64, 64)
    print("Evaluating Q=2 OOD Accuracy:")
    res_q2 = eval_model_ood(state_q2, 2)
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    lengths = sorted(res_q64.keys())
    
    plt.plot(lengths, [res_q64[l][1] for l in lengths], marker='x', color='r', linestyle='--', label="Q=64 Token Accuracy")
    plt.plot(lengths, [res_q2[l][1] for l in lengths], marker='o', color='b', linestyle='-', label="Q=2 Token Accuracy")
    
    plt.axvline(x=60, color='gray', linestyle='--', label="Max Train Length (60)")
    plt.title("Comparative OOD Generalization (Token Accuracy)", fontsize=14)
    plt.xlabel("String Length (Bits)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparative_ood_accuracy.pdf"))
    plt.close()
    
    print("Shortcut experiment completed successfully!")
