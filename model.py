# model.py
import jax
import jax.numpy as jnp
from flax import linen as nn
from constants import *
from stack_utils import update_stack

class StackMachineCell(nn.Module):
    """
    Single step of the machine.
    Wrapped in nn.scan to handle recurrence.
    """
    stack_depth: int = STACK_DEPTH
    
    @nn.compact
    def __call__(self, carry, inputs):
        
        # Unpack carry and inputs
        stack, ptr, r_prev, s_prev = carry
        x_t, true_act, true_s, use_forcing = inputs
        
        # 1. Controller Logic
        x_emb = nn.Embed(VOCAB_SIZE, HIDDEN_DIM)(x_t)
        s_emb = jax.nn.one_hot(s_prev, NUM_STATES)
        r_emb = jax.nn.one_hot(r_prev, STACK_VOCAB_SIZE)
        
        flat_input = jnp.concatenate([x_emb, s_emb, r_emb], axis=-1)
        
        logits_mem = nn.Dense(NUM_MEM_ACTIONS, name="head_mem")(flat_input)
        logits_buf = nn.Dense(NUM_BUF_ACTIONS, name="head_buf")(flat_input)
        logits_state = nn.Dense(NUM_STATES, name="head_state")(flat_input)
        
        # 2. Decide Actions (Teacher Forcing vs Model Prediction)
        
        
        pred_act = jnp.argmax(logits_mem, axis=-1)
        pred_state = jnp.argmax(logits_state, axis=-1)
        
        # use_forcing is broadcasted, so we select based on it
        action_to_exec = jnp.where(use_forcing > 0, true_act, pred_act)
        next_s = jnp.where(use_forcing > 0, true_s, pred_state)
        
        # 3. Update Stack (Vectorized)
        stack_new, ptr_new, r_new = jax.vmap(update_stack)(stack, ptr, action_to_exec)
        
        new_carry = (stack_new, ptr_new, r_new, next_s)
        outputs = (logits_mem, logits_buf, logits_state)
        
        return new_carry, outputs



class NeuralStackMachine(nn.Module):
    @nn.compact
    def __call__(self, x, true_actions, true_states, use_forcing):
        """
        x: [Batch, Seq]
        true_actions: [Batch, Seq] (Used if use_forcing=True)
        true_states: [Batch, Seq]  (Used if use_forcing=True)
        use_forcing: bool (Scalar)
        """
        batch_size, seq_len = x.shape
        
        # Initial Carry
        init_stack = jnp.zeros((batch_size, self.stack_depth), dtype=jnp.int32)
        init_ptr = jnp.zeros((batch_size,), dtype=jnp.int32)
        init_reg = jnp.zeros((batch_size,), dtype=jnp.int32)
        init_state = jnp.zeros((batch_size,), dtype=jnp.int32)
        carry = (init_stack, init_ptr, init_reg, init_state)
        
        # Broadcast forcing flag
        forcing_seq = jnp.full((batch_size, seq_len), use_forcing, dtype=jnp.int32)
        
        scan_inputs = (x, true_actions, true_states, forcing_seq)
        
        scan_layer = nn.scan(
            StackMachineCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1, 
            out_axes=1
        )
        
        final_carry, sequence_outputs = scan_layer()(carry, scan_inputs)
        
        return sequence_outputs