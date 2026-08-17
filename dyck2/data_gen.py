import numpy as np
import jax.numpy as jnp
from constants import *

def generate_dyck_string(max_depth, length):
    """
    Generates a Dyck-2 prefix of a given length.
    At each step depth should be >= 0!
    """
    prefix = []
    prefix_acts = []
    open_stack = []
    
    for _ in range(length):
        if len(open_stack) == 0:
            if np.random.rand() > 0.5:
                prefix.append(DYCK_OPEN1)
                prefix_acts.append(ACT_PUSH_0)
                open_stack.append(DYCK_OPEN1)
            else:
                prefix.append(DYCK_OPEN2)
                prefix_acts.append(ACT_PUSH_1)
                open_stack.append(DYCK_OPEN2)
        elif len(open_stack) >= max_depth:
            last_open = open_stack.pop()
            if last_open == DYCK_OPEN1:
                prefix.append(DYCK_CLOSE1)
            else:
                prefix.append(DYCK_CLOSE2)
            prefix_acts.append(ACT_POP)
        else:
            if np.random.rand() > 0.5:
                if np.random.rand() > 0.5:
                    prefix.append(DYCK_OPEN1)
                    prefix_acts.append(ACT_PUSH_0)
                    open_stack.append(DYCK_OPEN1)
                else:
                    prefix.append(DYCK_OPEN2)
                    prefix_acts.append(ACT_PUSH_1)
                    open_stack.append(DYCK_OPEN2)
            else:
                last_open = open_stack.pop()
                if last_open == DYCK_OPEN1:
                    prefix.append(DYCK_CLOSE1)
                else:
                    prefix.append(DYCK_CLOSE2)
                prefix_acts.append(ACT_POP)
                
    suffix = []
    suffix_acts = []
    while open_stack:
        last_open = open_stack.pop()
        if last_open == DYCK_OPEN1:
            suffix.append(DYCK_CLOSE1)
        else:
            suffix.append(DYCK_CLOSE2)
        suffix_acts.append(ACT_POP)
            
    return prefix, suffix, prefix_acts, suffix_acts

def generate_dyck_batch(batch_size, length):
    """
    Generates a batch of Dyck-2 traces for training
    """
    inputs = []
    targets = []
    masks = []
    target_acts = []
    act_masks = []
    
    # the max possible length is 2*length + 2
    total_len = 2 * length + 2
    
    for _ in range(batch_size):
        prefix, suffix, prefix_acts, suffix_acts = generate_dyck_string(STACK_DEPTH - 10, length)
        
        #[prefix] + [=] + [suffix] + [EOS]
        full_trace = prefix + [DYCK_EQ] + suffix + [DYCK_EOS]
        
        # Targets are shifted by 1 (next token prediction)
        # we use the standard autoregressive mask
        target = full_trace[1:] + [DYCK_PAD]
        
        # Mask only the suffix part (after the '=')
        mask = [0] * (len(prefix) + 1) + [1] * (len(suffix) + 1)
        
        t_acts = prefix_acts + [ACT_POP] + suffix_acts + [ACT_NOOP]
        a_mask = [1] * len(full_trace)
        
        # Pad
        pad_len = total_len - len(full_trace)
        full_trace += [DYCK_PAD] * pad_len
        target += [DYCK_PAD] * pad_len
        mask += [0] * pad_len
        t_acts += [ACT_NOOP] * pad_len
        a_mask += [0] * pad_len
        
        inputs.append(full_trace)
        targets.append(target)
        masks.append(mask)
        target_acts.append(t_acts)
        act_masks.append(a_mask)
        
    return (jnp.array(inputs), jnp.array(targets), jnp.array(masks), jnp.array(target_acts), jnp.array(act_masks))

def generate_fixed_batch(batch_size, length):
    """Utility for evaluation."""
    prefixes = []
    for _ in range(batch_size):
        prefix, _ = generate_dyck_string(STACK_DEPTH - 10, length)
        prefixes.append(prefix)
    return jnp.array(prefixes)
