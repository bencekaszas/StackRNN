import jax
import jax.numpy as jnp
from flax import linen as nn
from constants import *
from stack_utils import soft_update_stack

class StackRNNCell(nn.Module):
    """single step of the auto-regressive StackRNN."""
    stack_depth: int = STACK_DEPTH
    hard_actions: bool = False
    
    @nn.compact
    def __call__(self, carry, x_emb):
        stack, state_prev = carry
        
        stack_top = stack[:, 0]
        
        # Project inputs into hidden space
        state_emb = nn.Dense(HIDDEN_DIM, name="state_embed")(state_prev)
        stack_top_emb = nn.Dense(HIDDEN_DIM, name="stack_top_embed")(stack_top)         
        flat_input = jnp.concatenate([x_emb, state_emb, stack_top_emb], axis=-1)
        
        # Output heads
        logits_mem = nn.Dense(NUM_MEM_ACTIONS)(flat_input)
        logits_buf = nn.Dense(VOCAB_SIZE)(flat_input)
        logits_state = nn.Dense(NUM_STATES)(flat_input)
        
        # Action probabilities (still need to sum to 1 for soft-stack logic)
        action_probs = nn.softmax(logits_mem)
        
        if self.hard_actions:
            # Discrete actions at inference time
            max_act = jnp.argmax(action_probs, axis=-1)
            action_probs = jax.nn.one_hot(max_act, NUM_MEM_ACTIONS)
        
        # Vectorized stack update
        stack_new, _ = jax.vmap(soft_update_stack)(stack, action_probs)

        # State Update
        #next_state = nn.softmax(logits_state, axis=-1)
        next_state = jnp.tanh(logits_state)  # Using tanh activation for state representation
        
        new_carry = (stack_new, next_state)
        return new_carry, (logits_buf, action_probs)

class StackRNN(nn.Module):
    """An auto-regressive StackRNN model."""
    cell_cls = StackRNNCell
    use_one_hot_emb: bool = False
    
    @nn.compact
    def embed(self, x):
        """A separate method for embedding the input."""
        if self.use_one_hot_emb:
            return jax.nn.one_hot(x, VOCAB_SIZE)
        else:
            return nn.Embed(VOCAB_SIZE, HIDDEN_DIM, name="input_embed")(x)

    @nn.compact
    def __call__(self, x, hard_actions=False):
        batch_size, seq_len = x.shape
        x_emb = self.embed(x)
        
        if self.use_one_hot_emb:
            x_emb = nn.Dense(HIDDEN_DIM, name="input_proj")(x_emb)

        # Init Carry
        init_stack = jnp.zeros((batch_size, STACK_DEPTH, STACK_VOCAB_SIZE))
        init_stack = init_stack.at[:, :, STACK_NULL].set(1.0)
        init_state = jnp.zeros((batch_size, NUM_STATES), dtype=jnp.float32)
        carry = (init_stack, init_state)
        
        scan_layer = nn.scan(self.cell_cls, variable_broadcast="params", 
                             split_rngs={"params": False}, in_axes=1, out_axes=1)
        
        final_carry, (logits_buf, _) = scan_layer(hard_actions=hard_actions)(carry, x_emb)
        
        return logits_buf, final_carry
