# run_experiment.py
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
import numpy as np

from constants import *
from data_gen import generate_rev_trace
from models import NeuralStackMachine, NeuralSoftStackMachine, VanillaLSTM

def create_train_state(model_class, key):
    model = model_class()
    dummy_x = jnp.zeros((1, SEQ_LENGTH), dtype=jnp.int32)
    # Init with dummy inputs
    params = model.init(key, dummy_x, dummy_x, dummy_x, use_forcing=False)
    tx = optax.adam(LEARNING_RATE)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch, use_forcing):
    inputs, tgt_mem, tgt_buf, tgt_state = batch
    
    def loss_fn(params):
        logits = state.apply_fn(params, inputs, tgt_mem, tgt_state, use_forcing)
        l_mem, l_buf, l_state = logits
        
        # Main Buffer Loss
        loss_b = optax.softmax_cross_entropy_with_integer_labels(l_buf, tgt_buf).mean()
        
        # Auxiliary Losses (only if model outputs them, i.e., Stack Machines)
        # We check shape of logits to determine if it's a dummy output
        loss_m = 0.0
        loss_s = 0.0
        if l_mem.shape[-1] > 1:
            loss_m = optax.softmax_cross_entropy_with_integer_labels(l_mem, tgt_mem).mean()
            loss_s = optax.softmax_cross_entropy_with_integer_labels(l_state, tgt_state).mean()
            
        total_loss = loss_b + loss_m + loss_s
        acc = (jnp.argmax(l_buf, -1) == tgt_buf).mean()
        return total_loss, acc
        
    (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, acc

def evaluate(state, seq_len, n_samples=100):
    inputs, tgt_mem, tgt_buf, tgt_state = generate_rev_trace(n_samples, seq_len)
    
    # Inference (No forcing)
    logits = state.apply_fn(state.params, inputs, tgt_mem, tgt_state, False)
    pred_buf = jnp.argmax(logits[1], -1)
    
    # Accuracy: fraction of perfectly correct sequences
    # (Checking exact match on output buffer)
    # We ignore PAD tokens in accuracy? Usually for reverse task we check exact string match.
    # Here checking element-wise mean is simpler for training curves.
    token_acc = (pred_buf == tgt_buf).mean()
    return token_acc

def run(model_name, model_class):
    print(f"\n--- Training {model_name} ---")
    key = jax.random.PRNGKey(42)
    state = create_train_state(model_class, key)
    
    history = {"step": [], "id_acc": [], "ood_acc": []}
    
    for step in range(STEPS):
        batch = generate_rev_trace(BATCH_SIZE, SEQ_LENGTH)
        
        # Linear decay of teacher forcing from 100% to 0% over first 2000 steps
        forcing_prob = max(0.0, 1.0 - step / 2000.0)
        use_forcing = (jax.random.uniform(jax.random.PRNGKey(step)) < forcing_prob)
        
        state, loss, acc = train_step(state, batch, use_forcing)
        
        if step % EVAL_FREQ == 0:
            id_acc = evaluate(state, SEQ_LENGTH)
            ood_acc = evaluate(state, TEST_SEQ_LENGTH)
            
            history["step"].append(step)
            history["id_acc"].append(id_acc)
            history["ood_acc"].append(ood_acc)
            
            print(f"Step {step:4d} | ID Acc: {id_acc:.2%} | OOD Acc: {ood_acc:.2%} | Forcing: {forcing_prob:.1f}")
            
    return history

if __name__ == "__main__":
    results = {}
    results["Hard Stack"] = run("Hard Stack", NeuralStackMachine)
    results["Soft Stack"] = run("Soft Stack", NeuralSoftStackMachine)
    results["LSTM"] = run("Vanilla LSTM", VanillaLSTM)
    
    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    for name, hist in results.items():
        ax[0].plot(hist["step"], hist["id_acc"], label=name)
        ax[1].plot(hist["step"], hist["ood_acc"], label=name)
        
    ax[0].set_title(f"In-Distribution (Len {SEQ_LENGTH})")
    ax[1].set_title(f"Out-of-Distribution (Len {TEST_SEQ_LENGTH})")
    
    for a in ax:
        a.set_xlabel("Steps")
        a.set_ylabel("Accuracy")
        a.legend()
        a.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig("stack_benchmark.png")
    print("\nBenchmark complete. Saved to stack_benchmark.png")