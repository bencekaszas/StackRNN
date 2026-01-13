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



def soft_update_stack(stack, action_probs):
    """
    "soft" stack update (Shift-based / No Pointer)
    
    stack: [Depth, Stack_Vocab_Size]
    action_probs: [4] (NOOP, PUSH0, PUSH1, POP)
    """
    # actions
    p_noop = action_probs[ACT_NOOP]
    p_push0 = action_probs[ACT_PUSH_0]
    p_push1 = action_probs[ACT_PUSH_1]
    p_pop = action_probs[ACT_POP]
    
    total_push = p_push0 + p_push1
    
    # 1. SHIFT DOWN (PUSH Candidate)
    # stack[i] <- stack[i-1], stack[0] <- new_val
    stack_down = jnp.roll(stack, 1, axis=0)
    
    # Calculate push value (handle divide by zero)
    safe_push = jnp.where(total_push > 0, total_push, 1.0)
    val_push_0 = p_push0 / safe_push
    val_push_1 = p_push1 / safe_push
    
    write_vec = jnp.array([0.0, 1.0, 0.0]) * val_push_0 + \
                jnp.array([0.0, 0.0, 1.0]) * val_push_1
    
    stack_down = stack_down.at[0].set(write_vec)
    
    # 2. SHIFT UP (POP Candidate)
    # stack[i] <- stack[i+1], stack[-1] <- NULL
    stack_up = jnp.roll(stack, -1, axis=0)
    null_vec = jnp.array([1.0, 0.0, 0.0])
    stack_up = stack_up.at[-1].set(null_vec)
    
    # 3. COMBINE
    stack_new = (p_noop * stack) + \
                (total_push * stack_down) + \
                (p_pop * stack_up)
    
    # 4. READ (Peek at top)
    # If we pop, we consume the top value.
    r_t = stack[0] * p_pop + null_vec * (1.0 - p_pop)
    
    return stack_new, r_t