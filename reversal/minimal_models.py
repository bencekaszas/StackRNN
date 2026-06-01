import jax
import jax.numpy as jnp
from flax import linen as nn
import sys
import os

# Add root to path for stack_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stack_utils import soft_update_stack

class MinimalStackRNNCell(nn.Module):
    """
    A minimal StackRNN cell matching the Lemma 1 architecture:
    Input z_t = [r_t, x_t, q_{t-1}] (10 dimensions)
    """
    hard_actions: bool = False

    @nn.compact
    def __call__(self, carry, x_one_hot):
        stack, state_prev = carry
        
        # Lemma 1: r(t) is the distribution over the top of the stack
        r_t = stack[:, 0]
        
        # Basis: (r_null, r_0, r_1, x_pad, x_0, x_1, x_eq, x_eos, q_read, q_write)
        z_t = jnp.concatenate([r_t, x_one_hot, state_prev], axis=-1)
        
        logits_state = nn.Dense(2, use_bias=False, name="Wq")(z_t)
        logits_mem = nn.Dense(4, use_bias=False, name="Wa")(z_t)
        logits_buf = nn.Dense(5, use_bias=False, name="Wb")(z_t)
        
        action_probs = nn.softmax(logits_mem)
        if self.hard_actions:
            max_act = jnp.argmax(action_probs, axis=-1)
            action_probs = jax.nn.one_hot(max_act, 4)
        
        # Soft stack update
        stack_new, _ = jax.vmap(soft_update_stack)(stack, action_probs)

        # State Update
        next_state = nn.softmax(logits_state, axis=-1)
        
        new_carry = (stack_new, next_state)
        return new_carry, (logits_buf, action_probs)

class MinimalStackRNN(nn.Module):
    """Minimal model for reversal task."""
    stack_depth: int = 100

    @nn.compact
    def __call__(self, x, hard_actions=False):
        batch_size, seq_len = x.shape
        # x is indices, convert to one-hot (5 dims)
        x_one_hot = jax.nn.one_hot(x, 5)
        
        # Init Carry
        init_stack = jnp.zeros((batch_size, self.stack_depth, 3))
        init_stack = init_stack.at[:, :, 0].set(1.0) # NULL=1
        
        # Initial state q(0) = READ (Basis: [1, 0])
        init_state = jnp.zeros((batch_size, 2))
        init_state = init_state.at[:, 0].set(1.0)
        
        carry = (init_stack, init_state)
        
        scan_layer = nn.scan(
            MinimalStackRNNCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1
        )
        
        final_carry, (logits_buf, _) = scan_layer(hard_actions=hard_actions)(carry, x_one_hot)
        return logits_buf, final_carry
