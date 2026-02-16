# data_gen.py
import numpy as np
import jax.numpy as jnp
from constants import *

def generate_rev_trace(batch_size, seq_length=SEQ_LENGTH):
    """
    Generates data for auto-regressive training of the reverse task.
    
    Returns:
        - inputs: The full sequence, e.g., `[1, 0, 1, =, 1, 0, 1, EOS, PAD, ...]`
        - targets: The sequence shifted by one, which the model must predict.
        - mask: A mask to apply to the loss, so it's only calculated on the output part.
    """
    # The total length needs to accommodate input + '=' + output + EOS
    # Max input length is seq_length. Max output is seq_length + 1 (for EOS).
    total_len = 2 * seq_length + 2

    inputs = np.full((batch_size, total_len), VOCAB_PAD, dtype=np.int32)
    targets = np.full((batch_size, total_len), VOCAB_PAD, dtype=np.int32)
    mask = np.zeros((batch_size, total_len), dtype=np.float32)

    lengths = np.random.randint(1, seq_length + 1, size=batch_size)

    for i in range(batch_size):
        L = lengths[i]
        bits = np.random.randint(VOCAB_0, VOCAB_1 + 1, size=L)
        
        # --- Create the full sequence ---
        prompt = np.concatenate([bits, [VOCAB_EQ]])
        reversed_bits = bits[::-1]
        # The output includes the EOS token from the *buffer actions* vocabulary
        output = np.concatenate([reversed_bits, [VOCAB_EOS]])
        
        full_seq = np.concatenate([prompt, output])
        # --- Create inputs and targets for auto-regressive training ---
        # Input is the full sequence (except the last element)
        inputs[i, :len(full_seq)-1] = full_seq[:-1]
        # Target is the full sequence shifted left by one
        targets[i, :len(full_seq)-1] = full_seq[1:]
        
        # --- Create the loss mask ---
        # We only want to calculate the loss on the output part of the sequence.
        # The loss for predicting the first reversed bit starts at index L,
        # where the model has just seen the '=' sign.
        loss_start_index = L
        loss_end_index = L + len(output) # up to and including the EOS token
        mask[i, loss_start_index:loss_end_index] = 1.0
            
    return jnp.array(inputs), jnp.array(targets), jnp.array(mask)


def generate_fixed_batch(batch_size, target_length):
    """
    Generates a batch of prompts for inference, where every sequence has 
    exactly `target_length` bits.
    """
    # The total sequence length needed for Input + Eq
    total_seq_len = target_length + 1
    
    prompts = np.full((batch_size, total_seq_len), VOCAB_PAD, dtype=np.int32)
    
    for i in range(batch_size):
        L = target_length
        bits = np.random.randint(VOCAB_0, VOCAB_1 + 1, size=L)
        
        # Prompt is just bits + '='
        prompts[i, :L] = bits
        prompts[i, L] = VOCAB_EQ
            
    return jnp.array(prompts)



def generate_rev_trace_old(batch_size, seq_length=SEQ_LENGTH):
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