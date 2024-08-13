import jax
from jax import numpy as jnp
from jax import tree_util as jtu
import chex
import optax
from optax import Updates, ScalarOrSchedule, GradientTransformation
from typing import Any, NamedTuple, Union
import scheduler

ScalarOrPytree = Union[float, Any]

class SGDState(NamedTuple):
    count: chex.Array

def sgd(
    learning_rate: optax.ScalarOrSchedule
) -> GradientTransformation:
    """Simple SGD without momentum.
    
    Updates x_{t+1} = x_t - eta_t * (g_t + mu * x_t),
    where eta_t is the learning rate and mu is the weight decay constant.

    Args:
        learning_rate: The learning rate scheduler.
        weight_decay (float): The weight decay constant. Defaults to 0.

    Returns:
        A `GradientTransformation` object.
    """
    
    def init_fn(params):
        del params  # deletes params - not used
        return SGDState(count=jnp.zeros([], jnp.int32))  # returns SGDState with count set to zero
    
    def update_fn(updates, state, params):  # define update function
        if callable(learning_rate): #is it callable (schedule) or scalar
            lr = learning_rate(state.count)  # get learning rate if it's a schedule
        else:
            lr = learning_rate
        
        ogd_updates = jtu.tree_map(lambda g: -lr * g, updates)  # computes the OGD updates
        count_inc = optax.safe_int32_increment(state.count)  # safely increments the count (avoids overflow)
        return ogd_updates, SGDState(count=count_inc)  # returns OGD updates and the new state
    
    
    return GradientTransformation(init_fn, update_fn)
