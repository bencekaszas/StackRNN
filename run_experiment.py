import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import linen as nn
import optax
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from functools import partial



from constants import *
from data_gen import generate_rev_trace, generate_fixed_batch
from models import StackRNN

def create_train_state(model, key, learning_rate, dummy_input):
    params = model.init(key, dummy_input)['params']
    tx = optax.chain(
        optax.adam(learning_rate)
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def masked_loss(logits, targets, mask):
    """Masked softmax cross-entropy loss."""
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    masked_loss = loss * mask
    return masked_loss.sum() / jnp.maximum(mask.sum(), 1e-9)

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

# def get_initial_carry(state, prompt):
#     batch_size = prompt.shape[0]

#     init_stack = jnp.zeros((batch_size, STACK_DEPTH, STACK_VOCAB_SIZE))
#     init_stack = init_stack.at[:, :, STACK_NULL].set(1.0)
#     init_state = jnp.zeros((batch_size, NUM_STATES), dtype=jnp.float32)
#     carry = (init_stack, init_state)
    
#     prompt_emb = nn.Embed(VOCAB_SIZE, HIDDEN_DIM, name="input_embed").apply({'params': state.params['input_embed']}, prompt)
    
#     cell = StackRNN.cell_cls()
#     for i in range(prompt.shape[1]):
#         x_emb_step = prompt_emb[:, i]
#         carry, _ = cell.apply({'params': state.params['ScanStackRNNCell_0']}, carry, x_emb_step)

#     return carry

def evaluate(state, prompt, max_len=100):
    """Auto-regressive decoding with efficient state passing."""
    _, carry = state.apply_fn({'params': state.params}, x=prompt)

    #carry = (carry[0][:, -1:], carry[1])  

    decoder_input = jnp.full((prompt.shape[0], 1), VOCAB_EQ, dtype=jnp.int32)
    generated_sequence = []
    
    cell = StackRNN.cell_cls()
    embed_params = state.params['input_embed']

    for _ in range(max_len):
        decoder_emb = nn.Embed(VOCAB_SIZE, HIDDEN_DIM, name="input_embed").apply({'params': embed_params}, decoder_input)
        
        carry, logits = cell.apply({'params': state.params['ScanStackRNNCell_0']}, carry, decoder_emb[:, 0])
        
        next_token = jnp.argmax(logits, axis=-1)
        generated_sequence.append(next_token)
        
        if (next_token == VOCAB_EOS).all():
            break

        decoder_input = next_token[:, None]

    return jnp.concatenate(generated_sequence, axis=0)




def evaluate_recursive(state, prompt, max_len=100):
    """Auto-regressive decoding."""
    generated_sequence = prompt
    
    for _ in range(max_len):
        logits, _ = state.apply_fn({'params': state.params}, x=generated_sequence)
        next_token_logits = logits[:, -1, :] # Logits for the next token
        next_token = jnp.argmax(next_token_logits, axis=-1)
        
        if next_token[0] == VOCAB_EOS:
            break
            
        generated_sequence = jnp.concatenate([generated_sequence, next_token[:, None]], axis=1)
        
    return generated_sequence





if __name__ == "__main__":
    model = StackRNN()
    
    TEST_LENGTHS = [10, 20, 40, 60, 70, 80, 100, 120, 140]
    TRAINING_SEQ_LEN = SEQ_LENGTH
    N_EVAL_SAMPLES = 100

    final_results = {}

    # Main loop
    print(f"\n=== Training StackRNN ===")
    
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((1, 2 * TRAINING_SEQ_LEN + 2), dtype=jnp.int32)
    state = create_train_state(model, key, LEARNING_RATE, dummy_input)

    print(state.params.keys())


    #Train Loop
    losses = []
    accs = []
    for step in range(STEPS + 1):
        rand_max_len = np.random.randint(10, TRAINING_SEQ_LEN + 1)
        batch = generate_rev_trace(BATCH_SIZE, rand_max_len)
        state, loss, acc = train_step(state, batch)
        losses.append(loss)
        accs.append(acc)
        if step % 500 == 0:
            print(f"Step {step} | Train Loss: {loss:.4f} | Train Acc: {acc:.2%}")

    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig("training_loss_curve.png")

    plt.figure(figsize=(10, 6))
    plt.plot(accs)
    plt.title("Training Accuracy Curve")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.savefig("training_accuracy_curve.png")



    
    #evaluate OOD
    print(f"--- Evaluating OOD for StackRNN ---")
    seq_accuracies = []
    token_accuracies = []
    
    for L in TEST_LENGTHS:
        prompts = generate_fixed_batch(N_EVAL_SAMPLES, L)
        correct_predictions = 0
        
        token_accuracies_l = []
        for i in range(N_EVAL_SAMPLES):
            prompt = prompts[i:i+1, :]
            prompt_bits = prompt[0, :L]


            #print(prompt)
            generated = evaluate(state, jnp.array(prompt_bits[None, :]), max_len=L+5)
            #generated2 = evaluate_recursive(state, prompt, max_len=L+5)
            

            generated_output = generated
            #generated_output2 = generated2[0]
            
            # Create ground truth output
            ground_truth = np.asarray(prompt_bits[::-1])
            ground_truth = np.concatenate([ground_truth, [VOCAB_EOS]])
            

            is_correct = False
            if len(generated_output) >= len(ground_truth):
                if np.array_equal(generated_output[:len(ground_truth)], ground_truth):
                    if len(generated_output) == len(ground_truth) or generated_output[len(ground_truth)] == VOCAB_EOS:
                        is_correct = True

            if is_correct:
                correct_predictions += 1
            
            #print(generated_output, ground_truth)

            # only check till the length of the ground truth for token accuracy
            if len(generated_output) > len(ground_truth):
                generated_output = generated_output[:len(ground_truth)]
            elif len(generated_output) < len(ground_truth):
                # Pad with EOS if generated output is shorter than ground truth
                generated_output = np.concatenate([generated_output, [VOCAB_EOS] * (len(ground_truth) - len(generated_output))])

            token_accuracy = (generated_output == ground_truth).mean()
            token_accuracies_l.append(token_accuracy)
                
        seq_acc = correct_predictions / N_EVAL_SAMPLES
        token_acc = np.mean(token_accuracies_l)
        seq_accuracies.append(seq_acc)
        token_accuracies.append(token_acc)
        print(f"Len {L}: Seq Acc: {seq_acc:.2%} | Token Acc: {token_acc:.2%}")
        
    final_results["StackRNN"] = (seq_accuracies, token_accuracies)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    plt.plot(TEST_LENGTHS, final_results["StackRNN"][0], marker='o')
    plt.plot(TEST_LENGTHS, final_results["StackRNN"][1], marker='x', label="Token Accuracy")
    plt.axvline(x=60, color='gray', linestyle='--', label="Max Train Length (60)")
    plt.title("OOD Generalization: Accuracy vs. String Length", fontsize=14)
    plt.xlabel("String Length (Bits)", fontsize=12)
    plt.ylabel("Sequence Accuracy (Exact Match)", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.xticks(TEST_LENGTHS)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("ood_generalization_plot.png")