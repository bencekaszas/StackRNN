# data_gen.py
import numpy as np
import jax.numpy as jnp
from constants import *

def generate_rev_trace(batch_size, seq_length=SEQ_LENGTH):
    """
    Generate data for the reverse task with full traces
    """

    lengths = np.random.randint(1, seq_length + 1, size=batch_size)
    seq_len = 2 * seq_length + 1  # input + = + output 

    inputs = np.full((batch_size, seq_len), VOCAB_PAD, dtype=np.int32)
    target_act = np.full((batch_size, seq_len), ACT_NOOP, dtype=np.int32)
    target_buf = np.full((batch_size, seq_len), OUT_NOOP, dtype=np.int32)
    target_state = np.full((batch_size, seq_len), STATE_READ, dtype=np.int32)
    
    
    for i in range(batch_size):
        L = lengths[i]
        bits = np.random.randint(1, 3, size=L)
        
        # read 
        # Input: bits + '='
        inputs[i, :L] = bits
        inputs[i, L] = VOCAB_EQ
        
        pop_start = L # start at equal!
        pop_end = pop_start + L
        
        target_act[i, :L] = bits
        
        # State
        target_state[i, :L] = STATE_READ
        target_state[i, L] = STATE_WRITE 
        
        # write
        target_act[i,pop_start : pop_end] = ACT_POP
        
        # buffer: Emit the popped bits 
        # DELAYED BY 1!!
        reversed_bits = bits[::-1]
        target_buf[i, pop_start+1 : pop_end+1] = reversed_bits 
        
        # State: stays in WRITE
        target_state[i, pop_start : pop_end] = STATE_WRITE

            
    return jnp.array(inputs), jnp.array(target_act), jnp.array(target_buf), jnp.array(target_state)



def generate_fixed_batch(batch_size, target_length):
    """
    Generates a batch where EVERY sequence has exactly `target_length` bits.
    """
    # Force lengths to be exactly target_length
    lengths = np.full(batch_size, target_length, dtype=np.int32)
    
    # Calculate the total sequence length needed for Input + Eq + Output
    # The original code uses 2 * seq_length + 1. 
    # We must ensure the array is large enough for the target_length.
    total_seq_len = 2 * target_length + 1
    
    inputs = np.full((batch_size, total_seq_len), VOCAB_PAD, dtype=np.int32)
    target_buf = np.full((batch_size, total_seq_len), OUT_NOOP, dtype=np.int32)
    
    # We don't really need target_act or target_state for unsupervised/eval 
    # but we create them to match the function signature
    target_act = np.full((batch_size, total_seq_len), ACT_NOOP, dtype=np.int32)
    target_state = np.full((batch_size, total_seq_len), STATE_READ, dtype=np.int32)

    for i in range(batch_size):
        L = lengths[i]
        bits = np.random.randint(1, 3, size=L)
        
        # Input: bits + '='
        inputs[i, :L] = bits
        inputs[i, L] = VOCAB_EQ
        
        pop_start = L
        pop_end = pop_start + L
        
        # Buffer: Output reversed bits
        reversed_bits = bits[::-1]
        target_buf[i, pop_start+1 : pop_end+1] = reversed_bits 
            
    return jnp.array(inputs), jnp.array(target_act), jnp.array(target_buf), jnp.array(target_state)