# data_gen.py
import numpy as np
import jax.numpy as jnp
from constants import *

def generate_rev_trace(batch_size, seq_length=SEQ_LENGTH):
    """
    Generates data for the REV task with full supervision traces.
    """
    # Random bit string lengths
    # seq_length is passed to determine max capacity, but we generate variable lengths
    # Max bits we can fit is roughly half the sequence length
    max_bits = (seq_length - 1) // 2
    lengths = np.random.randint(1, max_bits + 1, size=batch_size)
    
    inputs = np.full((batch_size, seq_length), VOCAB_PAD, dtype=np.int32)
    target_act = np.full((batch_size, seq_length), ACT_NOOP, dtype=np.int32)
    target_buf = np.full((batch_size, seq_length), OUT_NOOP, dtype=np.int32)
    target_state = np.full((batch_size, seq_length), STATE_READ, dtype=np.int32)
    
    for i in range(batch_size):
        L = lengths[i]
        bits = np.random.randint(1, 3, size=L) # 1 or 2
        
        # --- PHASE 1: READ ---
        inputs[i, :L] = bits
        inputs[i, L] = VOCAB_EQ
        
        target_act[i, :L] = bits # Push corresponding bits
        target_state[i, :L] = STATE_READ
        target_state[i, L] = STATE_WRITE # Switch at '='
        
        # --- PHASE 2: WRITE & POP ---
        pop_start = L + 1
        pop_end = pop_start + L
        
        # Pop actions (Delayed by 1 step from '=' arrival to allow state switch processing)
        # Note: In your notebook you started popping right after EQ, which is index L+1
        # Inputs during this phase are PAD (0)
        
        if pop_end <= seq_length: # Safety check
            target_act[i, pop_start : pop_end] = ACT_POP
            target_state[i, pop_start : pop_end] = STATE_WRITE
            
            # --- BUFFER OUPUT (DELAYED BY 1) ---
            # If we Pop at t, the result is available at t+1
            emit_start = pop_start + 1
            emit_end = emit_start + L
            
            if emit_end <= seq_length:
                reversed_bits = bits[::-1]
                target_buf[i, emit_start : emit_end] = reversed_bits
            
    return jnp.array(inputs), jnp.array(target_act), jnp.array(target_buf), jnp.array(target_state)