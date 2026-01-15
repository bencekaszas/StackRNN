import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
import numpy as np

from constants import *
from data_gen import generate_rev_trace
from models import NeuralStackMachine, NeuralSoftStackMachine, VanillaLSTM, SequentialStackMachine

def create_train_state(model_class, key):
    model = model_class()
    dummy_x = jnp.zeros((1, SEQ_LENGTH), dtype=jnp.int32)
    # Init with dummy inputs
    params = model.init(key, dummy_x, dummy_x, dummy_x, use_forcing=False)
    tx = optax.adam(LEARNING_RATE)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch, use_forcing, supervise_trace):
    inputs, tgt_mem, tgt_buf, tgt_state = batch
    
    def loss_fn(params):
        logits = state.apply_fn(params, inputs, tgt_mem, tgt_state, use_forcing)
        l_mem, l_buf, l_state = logits
        
        # Main Buffer Loss
        loss_b = optax.softmax_cross_entropy_with_integer_labels(l_buf, tgt_buf).mean()
        
        # Extra Losses (only if model outputs them)
        # Check shape of logits to determine if dummy
        loss_m = 0.0
        loss_s = 0.0
        if l_mem.shape[-1] > 1:
            loss_m = optax.softmax_cross_entropy_with_integer_labels(l_mem, tgt_mem).mean()
            loss_s = optax.softmax_cross_entropy_with_integer_labels(l_state, tgt_state).mean()
            
        total_loss = loss_b + (loss_m + loss_s) * supervise_trace
        acc = (jnp.argmax(l_buf, -1) == tgt_buf).mean()
        return total_loss, acc
        
    (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, acc

def evaluate(state, seq_len, n_samples=100):
    inputs, tgt_mem, tgt_buf, tgt_state = generate_rev_trace(n_samples, seq_len)
    
    # Inference
    logits = state.apply_fn(state.params, inputs, tgt_mem, tgt_state, False)
    pred_buf = jnp.argmax(logits[1], -1)
    
    # Accuracy: fraction of perfectly correct sequences
    # (Checking exact match on output buffer)
    # We ignore PAD tokens in accuracy? Usually for reverse task we check exact string match.
    token_acc = (pred_buf == tgt_buf).mean()
    return token_acc

def run(model_name, model_class, supervise_trace=True):
    print(f"\n--- Training {model_name} ---")
    key = jax.random.PRNGKey(42)
    state = create_train_state(model_class, key)
    
    history = {"step": [], "train_acc": [], "id_acc": [], "ood_acc": []}
    
    for step in range(STEPS):
        batch = generate_rev_trace(BATCH_SIZE, SEQ_LENGTH)
        
        trace_weight = 1.0 if supervise_trace else 0.0
        state, loss, acc = train_step(state, batch, use_forcing=False, supervise_trace=trace_weight)
        
        if step % EVAL_FREQ == 0:
            id_acc = evaluate(state, SEQ_LENGTH)
            ood_acc = evaluate(state, TEST_SEQ_LENGTH)
            
            history["step"].append(step)
            history["train_acc"].append(acc)
            history["id_acc"].append(id_acc)
            history["ood_acc"].append(ood_acc)
            
            print(f"Step {step:4d} | Train Acc: {acc:.2%} | ID Acc: {id_acc:.2%} | OOD Acc: {ood_acc:.2%}")
            
    return history

if __name__ == "__main__":
    results = {}
    results["Hard Stack"] = run("Hard Stack", NeuralStackMachine, supervise_trace=True)
    results["Hard Stack (Unsupervised)"] = run("Hard Stack (Unsupervised)", NeuralStackMachine, supervise_trace=False)
    results["Soft Stack (Supervised)"] = run("Soft Stack (Supervised)", NeuralSoftStackMachine, supervise_trace=True)
    results["Soft Stack (Unsupervised)"] = run("Soft Stack (Unsupervised)", NeuralSoftStackMachine, supervise_trace=False)
    results["LSTM"] = run("Vanilla LSTM", VanillaLSTM, supervise_trace=False)
    results["Seq Stack"] = run("Sequential", SequentialStackMachine, supervise_trace=True)
    results["Seq Stack (Unsupervised)"] = run("Sequential (Unsupervised)", SequentialStackMachine, supervise_trace=False)


    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    for name, hist in results.items():
        line, = ax[0].plot(hist["step"], hist["id_acc"], label=name)
        color = line.get_color()
        ax[0].plot(hist["step"], hist["train_acc"], linestyle="--", alpha=0.4, color=color)
        ax[1].plot(hist["step"], hist["ood_acc"], label=name, color=color)
        
    ax[0].set_title(f"In-Distribution (Len {SEQ_LENGTH})")
    ax[0].set_title(f"In-Distribution (Solid=Test, Dashed=Train)")
    ax[1].set_title(f"Out-of-Distribution (Len {TEST_SEQ_LENGTH})")
    
    for a in ax:
        a.set_xlabel("Steps")
        a.set_ylabel("Accuracy")
        a.legend()
        a.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig("stack_benchmark.png")