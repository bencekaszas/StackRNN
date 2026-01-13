import jax.numpy as jnp
from constants import *


def update_stack(stack, ptr, action):
    """
    stack update
    
    stack: [Depth]
    ptr: Scalar index
    action: Scalar action
    """
    # Actions:
    # NOOP: No change.
    # PUSH 0: stack[ptr] = STACK_0, ptr++
    # PUSH 1: stack[ptr] = STACK_1, ptr++
    # POP: val = stack[ptr-1], ptr--
    
    # READ
    pop_ptr = jnp.maximum(0, ptr - 1)
    popped_val = stack[pop_ptr]
    
    # WRITE
    push_val = action 
    
    # UPDATE
    is_push = (action == ACT_PUSH_0) | (action == ACT_PUSH_1)
    new_stack_push = stack.at[ptr].set(push_val)
    new_ptr_push = ptr + 1
    
    is_pop = (action == ACT_POP)
    new_ptr_pop = pop_ptr

    new_stack_pop = stack.at[pop_ptr].set(STACK_NULL) 
    
    # Combine
    stack = jnp.where(is_push, new_stack_push, stack)
    stack = jnp.where(is_pop, new_stack_pop, stack)
    
    ptr = jnp.where(is_push, new_ptr_push, ptr)
    ptr = jnp.where(is_pop, new_ptr_pop, ptr)
    
    r_t = jnp.where(is_pop, popped_val, STACK_NULL)
    
    return stack, ptr, r_t



def soft_update_stack(stack, ptr_dist, action_probs):
    """
    "soft" stack update
    
    stack: [Depth, Stack_Vocab_Size]
    ptr_dist: [Depth] (Distribution)
    action_probs: [4] (NOOP, PUSH0, PUSH1, POP)
    """
    # actions
    p_noop = action_probs[ACT_NOOP]
    p_push0 = action_probs[ACT_PUSH_0]
    p_push1 = action_probs[ACT_PUSH_1]
    p_pop = action_probs[ACT_POP]
    
    total_push = p_push0 + p_push1
    
    #READ
    # Shift pointer up
    pop_ptr_dist = jnp.roll(ptr_dist, -1)
    
    #boundary
    pop_ptr_dist = pop_ptr_dist.at[-1].set(0.0)
    pop_ptr_dist = pop_ptr_dist.at[0].add(ptr_dist[0])

    #weighted stack
    read_val_vec = jnp.sum(stack * pop_ptr_dist[:, None], axis=0)
    
    #WRITE
    val_push_0 = p_push0 / total_push
    val_push_1 = p_push1 / total_push
    

    write_vec = jnp.array([0.0, 1.0, 0.0]) * val_push_0 + jnp.array([0.0, 0.0, 1.0]) * val_push_1
    
    # mix old stack and new value
    write_gate = ptr_dist[:, None] * total_push
    stack_new = stack * (1.0 - write_gate) + write_vec[None, :] * write_gate
    
    # Erase
    pop_gate = pop_ptr_dist[:, None] * p_pop
    null_vec = jnp.array([1.0, 0.0, 0.0])
    stack_new = stack_new * (1.0 - pop_gate) + null_vec[None, :] * pop_gate
    
    #move ptr
    
    push_ptr_dist = jnp.roll(ptr_dist, 1)
    push_ptr_dist = push_ptr_dist.at[0].set(0.0) #bottom doesnt wrap to top
    
    # all movement possibilities
    new_ptr_dist = (p_noop * ptr_dist) + \
                   (total_push * push_ptr_dist) + \
                   (p_pop * pop_ptr_dist)
                   
    # Scale read value by probability of popping
    r_t = read_val_vec * p_pop
    
    return stack_new, new_ptr_dist, r_t