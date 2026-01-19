import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from constants import *
from data_gen import generate_rev_trace, generate_fixed_batch
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

#TODO: fix datagen with seq stack - we evaluate against shifted register
# add XOR task? for that models need nonlinearity i.e. Relu



if __name__ == "__main__":
    # --- Configuration ---
    MODELS_TO_RUN = {
        "Unsupervised Soft Stack": (NeuralSoftStackMachine, False), # (Class, supervise_trace)
        "Unsupervised Seq Stack": (SequentialStackMachine, False),
        "Vanilla LSTM": (VanillaLSTM, False)
    }

    TEST_LENGTHS = [10, 20, 40, 60, 70, 80, 100, 120, 140]
    TRAINING_SEQ_LEN = 60 # Standard training length
    N_EVAL_SAMPLES = 200  # Samples per length check

    final_results = {}

    # --- Main Loop ---
    for name, (model_cls, supervise) in MODELS_TO_RUN.items():
        print(f"\n=== Training {name} ===")
        
        # 1. Train
        # We use the standard training routine provided in your context
        # Assuming 'run' returns the training history and the final state
        # (Note: I'll adapt the 'run' logic slightly to return the trained state)
        
        key = jax.random.PRNGKey(42)
        state = create_train_state(model_cls, key)
        
        # Train Loop (Simplified from your run code)
        for step in range(2001): # 2000 steps
            batch = generate_rev_trace(BATCH_SIZE, TRAINING_SEQ_LEN)
            trace_weight = 1.0 if supervise else 0.0
            state, loss, acc = train_step(state, batch, use_forcing=False, supervise_trace=trace_weight)
            
            if step % 500 == 0:
                print(f"Step {step} | Train Acc: {acc:.2%}")

        # 2. Evaluate OOD (The specific graph you want)
        print(f"--- Evaluating OOD for {name} ---")
        accuracies = []
        
        for L in TEST_LENGTHS:
            # Generate FIXED length batch
            batch = generate_fixed_batch(N_EVAL_SAMPLES, L)
            inputs, tgt_mem, tgt_buf, tgt_state = batch
            
            # Run Inference
            # Note: We must allow JAX to re-compile for new shapes if necessary, 
            # or rely on dynamic shapes if enabled. 
            # Since 'state.apply_fn' is jitted inside 'train_step' but here we call it raw:
            logits = state.apply_fn(state.params, inputs, tgt_mem, tgt_state, False)
            
            # Calculate Accuracy
            pred_buf = jnp.argmax(logits[1], -1)
            
            # Strict accuracy: The whole sequence must match the target buffer
            # (excluding the initial PADs which are 0 in both)
            match = (pred_buf == tgt_buf)
            seq_acc = match.all(axis=1).mean()
            
            accuracies.append(seq_acc)
            print(f"Len {L}: {seq_acc:.2%}")
            
        final_results[name] = accuracies

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ['o', 's', '^']

    for i, (name, accs) in enumerate(final_results.items()):
        plt.plot(TEST_LENGTHS, accs, 
                label=name, 
                marker=markers[i], 
                linewidth=2.5, 
                markersize=8,
                alpha=0.8)

    # Add a vertical line to show where training distribution ends
    plt.axvline(x=60, color='gray', linestyle='--', alpha=0.6, label="Max Train Length (60)")

    plt.title("OOD Generalization: Accuracy vs. String Length", fontsize=14)
    plt.xlabel("String Length (Bits)", fontsize=12)
    plt.ylabel("Sequence Accuracy (Exact Match)", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.xticks(TEST_LENGTHS)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()