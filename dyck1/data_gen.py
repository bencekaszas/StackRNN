import numpy as np
import jax.numpy as jnp
from constants import *

def generate_dyck_string(max_depth, length):
    """
    Generates a Dyck-1 prefix of a given length
    at each step depth should be >= 0!
    """
    prefix = []
    current_depth = 0
    for _ in range(length):
        # Decide whether to open or close
        # If depth is 0, open only 
        if current_depth == 0:
            prefix.append(DYCK_OPEN)
            current_depth += 1
        elif current_depth >= max_depth:
            prefix.append(DYCK_CLOSE)
            current_depth -= 1
        else:
            if np.random.rand() > 0.5:
                prefix.append(DYCK_OPEN)
                current_depth += 1
            else:
                prefix.append(DYCK_CLOSE)
                current_depth -= 1
                
    suffix = [DYCK_CLOSE] * current_depth
    return prefix, suffix

def generate_dyck_batch(batch_size, length):
    """
    Generates a batch of Dyck-1 traces for training
    """
    inputs = []
    targets = []
    masks = []
    
    # the max possible length is 2*length + 2
    total_len = 2 * length + 2
    
    for _ in range(batch_size):
        prefix, suffix = generate_dyck_string(STACK_DEPTH - 10, length)
        
        #[prefix] + [=] + [suffix] + [EOS]
        full_trace = prefix + [DYCK_EQ] + suffix + [DYCK_EOS]
        
        # Targets are shifted by 1 (next token prediction)
        # we use the standard autoregressive mask
        target = full_trace[1:] + [DYCK_PAD]
        
        # Mask only the suffix part (after the '=')
        mask = [0] * (len(prefix) + 1) + [1] * (len(suffix) + 1)
        
        # Pad
        pad_len = total_len - len(full_trace)
        full_trace += [DYCK_PAD] * pad_len
        target += [DYCK_PAD] * pad_len
        mask += [0] * pad_len
        
        inputs.append(full_trace)
        targets.append(target)
        masks.append(mask)
        
    return (jnp.array(inputs), jnp.array(targets), jnp.array(masks))

def generate_fixed_batch(batch_size, length):
    """Utility for evaluation."""
    prefixes = []
    for _ in range(batch_size):
        prefix, _ = generate_dyck_string(STACK_DEPTH - 10, length)
        prefixes.append(prefix)
    return jnp.array(prefixes)
