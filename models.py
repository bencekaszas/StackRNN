# models.py
import jax
import jax.numpy as jnp
from flax import linen as nn
from constants import *
from stack_utils import update_stack, soft_update_stack

# --- 1. HARD STACK MACHINE ---
class StackMachineCell(nn.Module):
    stack_depth: int = STACK_DEPTH
    
    @nn.compact
    def __call__(self, carry, inputs):
        stack, ptr, r_prev, s_prev = carry
        x_t, true_act, true_s, use_forcing = inputs
        
        # Embedding
        x_emb = nn.Embed(VOCAB_SIZE, HIDDEN_DIM)(x_t)
        s_emb = jax.nn.one_hot(s_prev, NUM_STATES)
        r_emb = jax.nn.one_hot(r_prev, STACK_VOCAB_SIZE)
        
        flat_input = jnp.concatenate([x_emb, s_emb, r_emb], axis=-1)
        
        # Controller
        logits_mem = nn.Dense(NUM_MEM_ACTIONS)(flat_input)
        logits_buf = nn.Dense(NUM_BUF_ACTIONS)(flat_input)
        logits_state = nn.Dense(NUM_STATES)(flat_input)
        
        # Select Action
        pred_act = jnp.argmax(logits_mem, axis=-1)
        pred_state = jnp.argmax(logits_state, axis=-1)
        
        # Teacher Forcing
        action_to_exec = jnp.where(use_forcing > 0, true_act, pred_act)
        next_s = jnp.where(use_forcing > 0, true_s, pred_state)
        
        # Stack Update
        stack_new, ptr_new, r_new = jax.vmap(update_stack)(stack, ptr, action_to_exec)
        
        new_carry = (stack_new, ptr_new, r_new, next_s)
        return new_carry, (logits_mem, logits_buf, logits_state)

class NeuralStackMachine(nn.Module):
    @nn.compact
    def __call__(self, x, true_actions, true_states, use_forcing):
        batch_size, seq_len = x.shape
        
        # Init Carry
        init_stack = jnp.zeros((batch_size, STACK_DEPTH), dtype=jnp.int32)
        init_ptr = jnp.zeros((batch_size,), dtype=jnp.int32)
        init_reg = jnp.zeros((batch_size,), dtype=jnp.int32)
        init_state = jnp.zeros((batch_size,), dtype=jnp.int32)
        carry = (init_stack, init_ptr, init_reg, init_state)
        
        forcing_seq = jnp.full((batch_size, seq_len), use_forcing, dtype=jnp.int32)
        scan_inputs = (x, true_actions, true_states, forcing_seq)
        
        scan_layer = nn.scan(StackMachineCell, variable_broadcast="params", 
                             split_rngs={"params": False}, in_axes=1, out_axes=1)
        
        _, outputs = scan_layer()(carry, scan_inputs)
        return outputs # (logits_mem, logits_buf, logits_state)


# --- 2. SOFT STACK MACHINE ---
class SoftStackMachineCell(nn.Module):
    stack_depth: int = STACK_DEPTH
    
    @nn.compact
    def __call__(self, carry, inputs):
        stack, r_prev, s_prev = carry
        x_t, true_act, true_s, use_forcing = inputs
        
        # Embeddings (r_prev is now a vector, not an int index)
        x_emb = nn.Embed(VOCAB_SIZE, HIDDEN_DIM)(x_t)
        s_emb = jax.nn.one_hot(s_prev, NUM_STATES)
        r_emb = nn.Dense(HIDDEN_DIM)(r_prev) 
        
        flat_input = jnp.concatenate([x_emb, s_emb, r_emb], axis=-1)
        
        logits_mem = nn.Dense(NUM_MEM_ACTIONS)(flat_input)
        logits_buf = nn.Dense(NUM_BUF_ACTIONS)(flat_input)
        logits_state = nn.Dense(NUM_STATES)(flat_input)
        
        action_probs = nn.softmax(logits_mem)
        
        # Teacher Forcing (Mix true one-hot with predicted probs)
        true_act_onehot = jax.nn.one_hot(true_act, NUM_MEM_ACTIONS)
        forcing_gate = use_forcing[:, None]
        probs_to_exec = (forcing_gate * true_act_onehot) + ((1.0 - forcing_gate) * action_probs)
        
        pred_state = jnp.argmax(logits_state, axis=-1)
        next_s = jnp.where(use_forcing > 0, true_s, pred_state)
        
        stack_new, r_new = jax.vmap(soft_update_stack)(stack, probs_to_exec)
        
        new_carry = (stack_new, r_new, next_s)
        return new_carry, (logits_mem, logits_buf, logits_state)

class NeuralSoftStackMachine(nn.Module):
    @nn.compact
    def __call__(self, x, true_actions, true_states, use_forcing):
        batch_size, seq_len = x.shape
        
        init_stack = jnp.zeros((batch_size, STACK_DEPTH, STACK_VOCAB_SIZE))
        init_stack = init_stack.at[:, :, STACK_NULL].set(1.0)
        init_reg = jnp.zeros((batch_size, STACK_VOCAB_SIZE)) # Empty reg
        init_state = jnp.zeros((batch_size,), dtype=jnp.int32)
        
        carry = (init_stack, init_reg, init_state)
        forcing_seq = jnp.full((batch_size, seq_len), use_forcing, dtype=jnp.float32)
        
        scan_inputs = (x, true_actions, true_states, forcing_seq)
        
        scan_layer = nn.scan(SoftStackMachineCell, variable_broadcast="params", 
                             split_rngs={"params": False}, in_axes=1, out_axes=1)
        
        _, outputs = scan_layer()(carry, scan_inputs)
        return outputs


# --- 3. VANILLA RNN (LSTM) ---
class VanillaLSTM(nn.Module):
    @nn.compact
    def __call__(self, x, true_actions, true_states, use_forcing):
        # We ignore true_actions/states/forcing for the baseline 
        # as it just learns seq-to-seq mapping directly.
        
        batch_size, seq_len = x.shape
        
        # 1. Embed Input
        x_emb = nn.Embed(VOCAB_SIZE, HIDDEN_DIM)(x) # [B, T, H]
        
        # 2. LSTM Scan
        lstm_layer = nn.scan(
            nn.LSTMCell,
            variable_broadcast="params", 
            split_rngs={"params": False}, 
            in_axes=1, 
            out_axes=1
        )(features=HIDDEN_DIM)
        
        # Initial Carry: (c, h)
        dummy_cell = nn.LSTMCell(features=HIDDEN_DIM)
        carry = dummy_cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, HIDDEN_DIM))
        
        final_carry, hidden_states = lstm_layer(carry, x_emb)
        
        # 3. Output Head
        logits_buf = nn.Dense(NUM_BUF_ACTIONS)(hidden_states)
        
        # Return same structure as others for compatibility (mem/state logits are dummies)
        dummy_logits = jnp.zeros((batch_size, seq_len, 1)) 
        return dummy_logits, logits_buf, dummy_logits