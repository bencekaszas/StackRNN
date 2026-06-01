import jax
import jax.numpy as jnp
from minimal_models import MinimalStackRNNCell
from inspect_minimal import get_lemma_weights
from flax.core import unfreeze

BASIS_LABELS = ["r_NULL", "r_0", "r_1", "x_PAD", "x_0", "x_1", "x_EQ", "x_EOS", "q_READ", "q_WRITE"]

def final_debug():
    OMEGA = 5.0
    Wq, Wa, Wb = get_lemma_weights(omega=OMEGA)
    params = {'Wq': {'kernel': Wq.T}, 'Wa': {'kernel': Wa.T}, 'Wb': {'kernel': Wb.T}}
    cell = MinimalStackRNNCell(hard_actions=True)
    
    # Sequence: [x_0, EQ] -> expect [x_0, EOS]
    # Bit 0 is token 1. EQ is token 3.
    
    stack = jnp.zeros((1, 10, 3)).at[:, :, 0].set(1.0)
    state = jnp.zeros((1, 2)).at[:, 0].set(1.0)
    carry = (stack, state)
    
    print("--- TRACE [x_0, EQ] ---")
    for t, tok in enumerate([1, 3]):
        x_oh = jax.nn.one_hot(jnp.array([tok]), 5)
        
        # Basis Check
        r_t = carry[0][:, 0]
        z_t = jnp.concatenate([r_t, x_oh, carry[1]], axis=-1)[0]
        active = jnp.where(z_t > 0.5)[0]
        print(f"\nStep {t} | Token {tok} | Active Basis: {[BASIS_LABELS[i] for i in active]}")
        
        # Manual Logit Calc to verify indices
        print(f"  Logits Mem (NOOP, PUSH0, PUSH1, POP): {jnp.dot(z_t, Wa.T)}")
        print(f"  Logits State (READ, WRITE): {jnp.dot(z_t, Wq.T)}")
        
        carry, (logits_buf, action_probs) = cell.apply({'params': params}, carry, x_oh)
        print(f"  Action Taken: {jnp.argmax(action_probs[0])}")
        print(f"  Next State: {carry[1][0]}")
        print(f"  Buffer Logits: {logits_buf[0]}")

if __name__ == "__main__":
    final_debug()
