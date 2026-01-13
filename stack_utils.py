import jax.numpy as jnp
from constants import *


def update_stack(stack, ptr, action):
    """
    Functional stack update.
    stack: [Depth]
    ptr: Scalar index
    action: Scalar action
    """
    # 1. READ (Popping) logic (Pre-calculation)
    pop_ptr = jnp.maximum(0, ptr - 1)
    popped_val = stack[pop_ptr]
    
    # 2. WRITE (Pushing) logic
    # Map action to stack value: Action 1->Stack 1 (0), Action 2->Stack 2 (1)
    push_val = action 
    
    # 3. APPLY UPDATE
    is_push = (action == ACT_PUSH_0) | (action == ACT_PUSH_1)
    is_pop = (action == ACT_POP)
    
    # Calculate new states for Push scenario
    new_stack_push = stack.at[ptr].set(push_val)
    new_ptr_push = ptr + 1
    
    # Calculate new states for Pop scenario
    new_ptr_pop = pop_ptr
    new_stack_pop = stack.at[pop_ptr].set(STACK_NULL) 
    
    # Combine based on action
    stack = jnp.where(is_push, new_stack_push, stack)
    stack = jnp.where(is_pop, new_stack_pop, stack)
    
    ptr = jnp.where(is_push, new_ptr_push, ptr)
    ptr = jnp.where(is_pop, new_ptr_pop, ptr)
    
    # Output Register (r_t) - If we popped, r_t is popped_val. Otherwise NULL.
    r_t = jnp.where(is_pop, popped_val, STACK_NULL)
    
    return stack, ptr, r_t