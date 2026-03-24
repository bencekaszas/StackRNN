import jax
import jax.numpy as jnp
from flax import linen as nn
from constants import *
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stack_utils import soft_update_stack

class DyckStackRNNCell(nn.Module):
    """single step of the auto-regressive StackRNN for Dyck-1."""
    hard_actions: bool = False 

    @nn.compact
    def __call__(self, carry, x_emb):
        stack, state_prev = carry
        
        # Top of stack
        stack_top = stack[:, 0]
        
        state_emb = nn.Dense(HIDDEN_DIM, name="state_embed")(state_prev)
        stack_top_emb = nn.Dense(HIDDEN_DIM, name="stack_top_embed")(stack_top)         
        flat_input = jnp.concatenate([x_emb, state_emb, stack_top_emb], axis=-1)
        
        logits_mem = nn.Dense(NUM_MEM_ACTIONS)(flat_input)
        logits_buf = nn.Dense(DYCK_VOCAB_SIZE)(flat_input)
        logits_state = nn.Dense(DYCK_NUM_STATES)(flat_input)
        
        action_probs = nn.softmax(logits_mem)
        
        if self.hard_actions:
            max_act = jnp.argmax(action_probs, axis=-1)
            action_probs = jax.nn.one_hot(max_act, NUM_MEM_ACTIONS)
        
        stack_new, _ = jax.vmap(soft_update_stack)(stack, action_probs)

        next_state = nn.softmax(logits_state, axis=-1)
        new_carry = (stack_new, next_state)
        return new_carry, (logits_buf, action_probs)

class DyckStackRNN(nn.Module):
    """An auto-regressive StackRNN model for Dyck-1."""
    @nn.compact
    def __call__(self, x, hard_actions=False):
        batch_size, seq_len = x.shape
        x_emb = nn.Embed(DYCK_VOCAB_SIZE, HIDDEN_DIM, name="input_embed")(x)
        
        # Initialize Carry
        init_stack = jnp.zeros((batch_size, STACK_DEPTH, DYCK_STACK_VOCAB_SIZE))
        init_stack = init_stack.at[:, :, DYCK_STACK_NULL].set(1.0)
        init_state = jnp.zeros((batch_size, DYCK_NUM_STATES), dtype=jnp.float32)
        carry = (init_stack, init_state)
        
        # Consistent with models.py
        scan_layer = nn.scan(
            DyckStackRNNCell, 
            variable_broadcast="params", 
            split_rngs={"params": False}, 
            in_axes=1, 
            out_axes=1
        )
        
        # Pass hard_actions as a keyword argument to initialize the scanned cells
        final_carry, (logits_buf, _) = scan_layer(hard_actions=hard_actions)(carry, x_emb)
        
        return logits_buf, final_carry
