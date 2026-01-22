from abc import ABC, abstractmethod

import jax.numpy as jnp
import jax
from functools import partial
import time

class BaseEnergyClass(ABC):

    def __init__(self, config):
        self.n_bernoulli_features = config["n_bernoulli_features"]
        self.vmap_calculate_Energy_per_node = jax.vmap(self.calculate_Energy_per_node, in_axes=(None, 0, None))
        pass

    @abstractmethod
    def calculate_Energy(self):
        raise ValueError("not implemented")

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_per_node(self, H_graph, bins, node_graph_idx):
        pass

    @abstractmethod
    def calculate_relaxed_Energy(self):
        pass

    @abstractmethod
    def calculate_Energy_loss(self, H_graph, logits, node_gr_idx, Energy_func):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_CE(self, graphs, X_0, node_gr_idx):
        pass

    def get_log_p_0_from_energy(self, Energy, T):
        T = jnp.max(jnp.array([T, 10**-6]))

        log_p_0 = -1/T*Energy[...,0]
        return log_p_0