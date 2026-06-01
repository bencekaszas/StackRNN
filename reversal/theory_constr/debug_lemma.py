import jax
import jax.numpy as jnp
from minimal_models import MinimalStackRNN, MinimalStackRNNCell
from inspect_minimal import get_lemma_weights
from flax.core import unfreeze, freeze
import numpy as np

# Basis: (r_NULL, r_0, r_1, x_PAD, x_0, x_1, x_EQ, x_EOS, q_READ, q_WRITE)
BASIS_LABELS = [
    "r_NULL", "r_0", "r_1", 
    "x_PAD", "x_0", "x_1", "x_EQ", "x_EOS",
    "q_READ", "q_WRITE"
]

def debug_trace():
    model = MinimalStackRNN()
    key = jax.random.PRNGKey(42)
    params = model.init(key, jnp.zeros((1, 1), dtype=jnp.int32))['params']
    
    OMEGA = 10.0 # Use smaller omega for readable logit differences
    Wq_val, Wa_val, Wb_val = get_lemma_weights(omega=OMEGA)
    
    new_params = unfreeze(params)
    new_params['ScanMinimalStackRNNCell_0']['Wq']['kernel'] = Wq_val.T
    new_params['ScanMinimalStackRNNCell_0']['Wa']['kernel'] = Wa_val.T
    new_params['ScanMinimalStackRNNCell_0']['Wb']['kernel'] = Wb_val.T
    
    cell = MinimalStackRNNCell()
    cell_params = new_params['ScanMinimalStackRNNCell_0']
    
    # Trace: Input "0" (index 1), then "=" (index 3)
    # Expected: PUSH "0", then transition to WRITE and POP
    inputs = [1, 3] # [x_0, x_EQ]
    
    # Init Carry
    stack = jnp.zeros((1, 100, 3))
    stack = stack.at[:, :, 0].set(1.0) # NULL=1
    state = jnp.zeros((1, 2))
    state = state.at[:, 0].set(1.0) # q_READ=1
    carry = (stack, state)
    
    print("--- STEP-BY-STEP TRACE ---")
    for t, x_idx in enumerate(inputs):
        x_one_hot = jax.nn.one_hot(jnp.array([x_idx]), 5)
        
        # Manual z_t construction to verify basis
        r_t = carry[0][:, 0]
        z_t = jnp.concatenate([r_t, x_one_hot, carry[1]], axis=-1)
        
        print(f"\nStep {t} | Input Token: {x_idx}")
        print(f"z_t vector: {z_t[0]}")
        # Print non-zero indices of z_t
        active_z = jnp.where(z_t[0] > 0.5)[0]
        print(f"Active z_t indices: {active_z} {[BASIS_LABELS[i] for i in active_z]}")
        
        # Apply cell
        new_carry, (logits_buf, action_probs) = cell.apply({'params': cell_params}, carry, x_one_hot)
        
        print(f"Action Probs: {action_probs[0]} (NOOP, PUSH0, PUSH1, POP)")
        print(f"Next State: {new_carry[1][0]} (READ, WRITE)")
        print(f"Buffer Logits: {logits_buf[0]}")
        
        carry = new_carry

if __name__ == "__main__":
    debug_trace()
