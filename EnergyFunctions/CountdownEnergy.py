from .BaseEnergy import BaseEnergyClass
import jax
from functools import partial
import jax.numpy as jnp

class CountdownEnergyClass(BaseEnergyClass):

    def __init__(self, config):
        super().__init__(config)
        

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy(self, meta_graph, bins, node_gr_idx, A=1.0, B=1.2):
        total_energy = jnp.array([0.0], dtype=jnp.float32)
        breakdown = {"total_energy": total_energy}        
        return total_energy, breakdown, total_energy

    def calculate_relaxed_Energy(self, H_graph, bins, node_gr_idx, A=1.0, B=1.2):
        return self.calculate_Energy(H_graph, bins, node_gr_idx, A=A, B=B)

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_loss(self, H_graph, logits, node_gr_idx, A=1.0, B=1.2):
        return self.calculate_Energy(H_graph, logits, node_gr_idx, A=A, B=B)