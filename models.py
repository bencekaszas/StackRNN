import jax
import jax.numpy as jnp
from flax import linen as nn
from constants import *
from stack_utils import update_stack, soft_update_stack

MAX_LEN = 2 * TEST_SEQ_LENGTH + 2

class StackRNNCell(nn.Module):
    """single step of the auto-regressive StackRNN."""
    stack_depth: int = STACK_DEPTH
    
    @nn.compact
    def __call__(self, carry, x_emb):
        stack, state_prev = carry
        
        stack_top = stack[:, 0]
        
        #state_emb = jax.nn.one_hot(state_prev, NUM_STATES)
        state_emb = nn.Dense(HIDDEN_DIM, name="state_embed")(state_prev)
        stack_top_emb = nn.Dense(HIDDEN_DIM, name="stack_top_embed")(stack_top)         
        flat_input = jnp.concatenate([x_emb, state_emb, stack_top_emb], axis=-1)
        
        logits_mem = nn.Dense(NUM_MEM_ACTIONS)(flat_input)
        logits_buf = nn.Dense(VOCAB_SIZE)(flat_input)
        logits_state = nn.Dense(NUM_STATES)(flat_input)
        
        action_probs = nn.softmax(logits_mem)
        
        stack_new, _ = jax.vmap(soft_update_stack)(stack, action_probs)

        next_state = nn.softmax(logits_state, axis=-1)
        new_carry = (stack_new, next_state)
        return new_carry, logits_buf

class StackRNN(nn.Module):
    """An auto-regressive StackRNN model."""
    cell_cls = StackRNNCell
    
    @nn.compact
    def embed(self, x):
        """A separate method for embedding the input."""
        x_emb = nn.Embed(VOCAB_SIZE, HIDDEN_DIM, name="input_embed")(x)
        return x_emb

    @nn.compact
    def __call__(self, x):
        batch_size, seq_len = x.shape
        x_with_pos = self.embed(x)
        
        # Init Carry
        init_stack = jnp.zeros((batch_size, STACK_DEPTH, STACK_VOCAB_SIZE))
        init_stack = init_stack.at[:, :, STACK_NULL].set(1.0)
        init_state = jnp.zeros((batch_size,NUM_STATES), dtype=jnp.float32)
        carry = (init_stack, init_state)
        
        scan_layer = nn.scan(self.cell_cls, variable_broadcast="params", 
                             split_rngs={"params": False}, in_axes=1, out_axes=1)
        
        final_carry, logits_buf = scan_layer()(carry, x_with_pos)
        
        return logits_buf, final_carry









# --- Old Models---

class HardStackRNNCell_old(nn.Module):
    stack_depth: int = STACK_DEPTH
    ste_type: str = 'straight_through'
    
    @nn.compact
    def __call__(self, carry, inputs):
        stack, ptr, r_prev, state_prev = carry
        x_emb, true_act, true_s, use_forcing, gumbel_temperature = inputs
        
        state_emb = jax.nn.one_hot(state_prev, NUM_STATES)
        reg_emb = jax.nn.one_hot(r_prev, STACK_VOCAB_SIZE)
        
        stack_not_empty = (ptr > 0).astype(jnp.float32)[:, None]
        
        flat_input = jnp.concatenate([x_emb, state_emb, reg_emb, stack_not_empty], axis=-1)
        
        logits_mem = nn.Dense(NUM_MEM_ACTIONS)(flat_input)
        
        if self.ste_type == 'straight_through':
            y_soft = nn.softmax(logits_mem, axis=-1)
            pred_act_hard = jnp.argmax(logits_mem, axis=-1)
        elif self.ste_type == 'gumbel_softmax':
            gumbel_key = self.make_rng('gumbel')
            gumbel_noise = jax.random.gumbel(gumbel_key, logits_mem.shape)
            temp = gumbel_temperature[:, None]
            y_soft = nn.softmax((logits_mem + gumbel_noise) / temp)
            pred_act_hard = jnp.argmax(y_soft, axis=-1)
        else:
            raise ValueError(f"Unknown STE type: {self.ste_type}")

        y_hard = jax.nn.one_hot(pred_act_hard, logits_mem.shape[-1])
        pred_act_ste = y_soft + jax.lax.stop_gradient(y_hard - y_soft)
        
        logits_mem = pred_act_ste + jax.lax.stop_gradient(logits_mem - pred_act_ste)

        stack_new, ptr_new, r_new = jax.vmap(update_stack)(stack, ptr, pred_act_hard)
        
        r_new_emb = jax.nn.one_hot(r_new, STACK_VOCAB_SIZE)

        flat_input_rest = jnp.concatenate([x_emb, state_emb, r_new_emb], axis=-1)

        logits_buf = nn.Dense(NUM_BUF_ACTIONS)(flat_input_rest)
        logits_state = nn.Dense(NUM_STATES)(flat_input_rest)
        
        next_state = jnp.argmax(logits_state, axis=-1)
        
        new_carry = (stack_new, ptr_new, r_new, next_state)
        return new_carry, (logits_mem, logits_buf, logits_state)

class HardStackRNN_old(nn.Module):
    ste_type: str = 'straight_through'

    @nn.compact
    def __call__(self, x, true_actions, true_states, use_forcing, gumbel_temperature=1.0):
        batch_size, seq_len = x.shape
        
        x_emb = nn.Embed(VOCAB_SIZE, HIDDEN_DIM, name="input_embed")(x)
        positions = jnp.arange(seq_len)
        pos_emb = nn.Embed(num_embeddings=MAX_LEN, features=HIDDEN_DIM, name="position_embed")(positions)
        x_with_pos = x_emb + pos_emb
        
        init_stack = jnp.zeros((batch_size, STACK_DEPTH), dtype=jnp.int32)
        init_ptr = jnp.zeros((batch_size,), dtype=jnp.int32)
        init_reg = jnp.zeros((batch_size,), dtype=jnp.int32)
        init_state = jnp.zeros((batch_size,), dtype=jnp.int32)
        carry = (init_stack, init_ptr, init_reg, init_state)
        
        forcing_seq = jnp.full((batch_size, seq_len), use_forcing, dtype=jnp.int32)
        temp_seq = jnp.full((batch_size, seq_len), gumbel_temperature, dtype=jnp.float32)

        scan_inputs = (x_with_pos, true_actions, true_states, forcing_seq, temp_seq)
        
        scan_layer = nn.scan(
            HardStackRNNCell_old, 
            variable_broadcast="params", 
            split_rngs={'params': False, 'gumbel': True}, 
            in_axes=1, 
            out_axes=1,
            )
        
        _, outputs = scan_layer(ste_type=self.ste_type)(carry, scan_inputs)
        return outputs


class SoftStackRNNCell_old(nn.Module):
    stack_depth: int = STACK_DEPTH
    
    @nn.compact
    def __call__(self, carry, inputs):
        stack, r_prev, state_prev = carry
        x_emb, true_act, true_s, use_forcing, _ = inputs
        
        state_emb = jax.nn.one_hot(state_prev, NUM_STATES)
        r_emb = nn.Dense(HIDDEN_DIM)(r_prev) 
        
        stack_not_empty = (1.0 - stack[:, 0, STACK_NULL])[:, None]
        
        flat_input = jnp.concatenate([x_emb, state_emb, r_emb, stack_not_empty], axis=-1)
        
        logits_mem = nn.Dense(NUM_MEM_ACTIONS)(flat_input)
        logits_buf = nn.Dense(NUM_BUF_ACTIONS)(flat_input)
        logits_state = nn.Dense(NUM_STATES)(flat_input)
        
        action_probs = nn.softmax(logits_mem)
        
        stack_new, r_new = jax.vmap(soft_update_stack)(stack, action_probs)

        next_state = jnp.argmax(logits_state, axis=-1)
        
        new_carry = (stack_new, r_new, next_state)
        return new_carry, (logits_mem, logits_buf, logits_state)

class SoftStackRNN_old(nn.Module):
    @nn.compact
    def __call__(self, x, true_actions, true_states, use_forcing, gumbel_temperature=1.0):
        batch_size, seq_len = x.shape
        
        x_emb = nn.Embed(VOCAB_SIZE, HIDDEN_DIM, name="input_embed")(x)
        positions = jnp.arange(seq_len)
        pos_emb = nn.Embed(num_embeddings=MAX_LEN, features=HIDDEN_DIM, name="position_embed")(positions)
        x_with_pos = x_emb + pos_emb
        
        init_stack = jnp.zeros((batch_size, STACK_DEPTH, STACK_VOCAB_SIZE))
        init_stack = init_stack.at[:, :, STACK_NULL].set(1.0)
        init_reg = jnp.zeros((batch_size, STACK_VOCAB_SIZE)) 
        init_state = jnp.zeros((batch_size,), dtype=jnp.int32)
        
        carry = (init_stack, init_reg, init_state)
        forcing_seq = jnp.full((batch_size, seq_len), use_forcing, dtype=jnp.float32)
        temp_seq = jnp.full((batch_size, seq_len), gumbel_temperature, dtype=jnp.float32)

        scan_inputs = (x_with_pos, true_actions, true_states, forcing_seq, temp_seq)
        
        scan_layer = nn.scan(SoftStackRNNCell_old, variable_broadcast="params", 
                             split_rngs={"params": False}, in_axes=1, out_axes=1)
        
        _, outputs = scan_layer()(carry, scan_inputs)
        return outputs

class VanillaLSTM_old(nn.Module):
    @nn.compact
    def __call__(self, x, true_actions, true_states, use_forcing, gumbel_temperature=1.0):
        batch_size, seq_len = x.shape
        
        x_emb = nn.Embed(VOCAB_SIZE, HIDDEN_DIM)(x)
        positions = jnp.arange(seq_len)
        pos_emb = nn.Embed(num_embeddings=MAX_LEN, features=HIDDEN_DIM, name="position_embed")(positions)
        x_with_pos = x_emb + pos_emb
        
        lstm_layer = nn.scan(
            nn.LSTMCell,
            variable_broadcast="params", 
            split_rngs={"params": False}, 
            in_axes=1, 
            out_axes=1
        )(features=HIDDEN_DIM)
        
        dummy_cell = nn.LSTMCell(features=HIDDEN_DIM)
        carry = dummy_cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, HIDDEN_DIM))
        
        _, hidden_states = lstm_layer(carry, x_with_pos)
        
        logits_buf = nn.Dense(NUM_BUF_ACTIONS)(hidden_states)
        
        dummy_logits = jnp.zeros((batch_size, seq_len, 1)) 
        return dummy_logits, logits_buf, dummy_logits

class Transformer_old(nn.Module):
    num_heads: int = 4
    num_layers: int = 2
    qkv_dim: int = HIDDEN_DIM
    
    @nn.compact
    def __call__(self, x, true_actions, true_states, use_forcing, gumbel_temperature=1.0):
        batch_size, seq_len = x.shape
        
        x_emb = nn.Embed(VOCAB_SIZE, HIDDEN_DIM)(x)
        positions = jnp.arange(seq_len)
        pos_emb = nn.Embed(num_embeddings=MAX_LEN, features=HIDDEN_DIM, name="position_embed")(positions)
        x_with_pos = x_emb + pos_emb
        
        for _ in range(self.num_layers):
            x_attn = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.qkv_dim)(x_with_pos)
            x_with_pos = nn.LayerNorm()(x_with_pos + x_attn)
            
            x_ff = nn.Dense(features=self.qkv_dim * 2)(x_with_pos)
            x_ff = nn.relu(x_ff)
            x_ff = nn.Dense(features=self.qkv_dim)(x_ff)
            x_with_pos = nn.LayerNorm()(x_with_pos + x_ff)

        logits_buf = nn.Dense(NUM_BUF_ACTIONS)(x_with_pos)
        
        dummy_logits = jnp.zeros((batch_size, seq_len, 1))
        
        return dummy_logits, logits_buf, dummy_logits