import time
from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp


class Base(ABC):
    def __init__(self, config, EnergyClass, NoiseClass, model):
        ### TODO implement learning rate schedule correctly
        self.config = config
        self.bfloat16 = self.config["bfloat16"]
        self.N_test_basis_states = self.config["n_test_basis_states"]
        self.opt_update = None
        self.EnergyClass = EnergyClass
        self.NoiseDistrClass = NoiseClass
        self.model = model
        self.problem_name = self.config["problem_name"]
        self.dataset_name = self.config["dataset_name"]
        self.eval_step_factor = self.config["eval_step_factor"]
        self.n_sampling_rounds = self.config["n_sampling_rounds"]
        self.sampling_temp = self.config["sampling_temp"]
        self.ownership_weight = self.config.get("ownership_weight", 1.0)
        print("EVAL STEP FACTOR is", self.eval_step_factor)

        def _make_one_step_det(params, graphs, X_prev, energy_per_node, t_idx_per_node, key, step=0):
            return self.model.make_one_step(
                params, graphs, X_prev, energy_per_node, t_idx_per_node, key, deterministic=True, step=step
            )

        def _make_one_step_stoch(params, graphs, X_prev, energy_per_node, t_idx_per_node, key, step=0):
            return self.model.make_one_step(
                params, graphs, X_prev, energy_per_node, t_idx_per_node, key, deterministic=False, step=step
            )
        
        self._vmapped_make_one_step_det = jax.vmap(
            _make_one_step_det,
            in_axes=(None, None, 1, 1, None, 0, None),
            out_axes=(1, 0),
        )
        self._vmapped_make_one_step_stoch = jax.vmap(
            _make_one_step_stoch,
            in_axes=(None, None, 1, 1, None, 0, None),
            out_axes=(1, 0),
        )


        def _call_vmapped_make_one_step(params, graphs, X_prev, energy_per_node,t_idx_per_node, key, deterministic=True, step=0):
            if deterministic:
                return self._vmapped_make_one_step_det(params, graphs, X_prev, energy_per_node, t_idx_per_node, key, step)
            return self._vmapped_make_one_step_stoch(params, graphs, X_prev, energy_per_node, t_idx_per_node, key, step)

        self.vmapped_make_one_step = _call_vmapped_make_one_step

        self.NoiseDistrClass = NoiseClass
        self.Noise_func = self.NoiseDistrClass.calc_noise_loss
        self.beta_arr = self.NoiseDistrClass.beta_arr
        self.calc_loss = self.NoiseDistrClass.combine_losses

        self.loss_grad = jax.jit(jax.value_and_grad(self.get_loss, has_aux=True))
        self.pmap_sample = jax.pmap(self.sample, in_axes=(0, 0, None, 0))

        self.pmap_loss_backward = jax.pmap(self.loss_backward, in_axes=(0, 0, 0, None, 0), axis_name="device")

        self.vmapped_sample_forward_diff_process = jax.vmap(self.NoiseDistrClass.sample_forward_diff_process, in_axes=(1, None, 0), out_axes=(1,1, 0))

        self.relaxed_energy = EnergyClass.calculate_Energy
        self.relaxed_Energy_for_Loss = EnergyClass.calculate_Energy_loss

        def _relaxed_energy(meta_graph, bins, node_gr_idx, ownership_weight):
            if self.problem_name == "HCP":
                return self.relaxed_energy(meta_graph, bins, node_gr_idx, A=ownership_weight)
            return self.relaxed_energy(meta_graph, bins, node_gr_idx)

        def _relaxed_energy_loss(meta_graph, logits, node_gr_idx, ownership_weight):
            if self.problem_name == "HCP":
                return self.relaxed_Energy_for_Loss(meta_graph, logits, node_gr_idx, A=ownership_weight)
            return self.relaxed_Energy_for_Loss(meta_graph, logits, node_gr_idx)

        _vmapped_relaxed_energy = jax.vmap(_relaxed_energy, in_axes=(None, 1, None, None), out_axes=(1, 1, 1))
        _vmapped_relaxed_energy_for_Loss = jax.vmap(_relaxed_energy_loss, in_axes=(None, 1, None, None),
                                                        out_axes=(1))

        def vmapped_relaxed_energy(meta_graph, bins, node_gr_idx, ownership_weight=None):
            weight = self.ownership_weight if ownership_weight is None else ownership_weight
            return _vmapped_relaxed_energy(meta_graph, bins, node_gr_idx, weight)

        def vmapped_relaxed_energy_for_Loss(meta_graph, logits, node_gr_idx, ownership_weight=None):
            weight = self.ownership_weight if ownership_weight is None else ownership_weight
            return _vmapped_relaxed_energy_for_Loss(meta_graph, logits, node_gr_idx, weight)

        self.vmapped_relaxed_energy = vmapped_relaxed_energy

        self.vmapped_relaxed_energy_for_Loss = vmapped_relaxed_energy_for_Loss

        self.n_diffusion_steps = self.config["n_diffusion_steps"]
        self.N_basis_states = self.config["N_basis_states"]
        self.batch_size = self.config["batch_size"]



    @abstractmethod
    def get_loss(self):
        pass

    @abstractmethod
    def sample(self):
        pass

    def _reverse_KL_loss_value(self, log_q_0_T, log_p_0_T, diff_step_axis = 0):

        loss = jnp.mean(jnp.sum(log_q_0_T, axis = diff_step_axis) - jnp.sum(log_p_0_T , axis = diff_step_axis))

        return loss

    def _forward_KL_loss_value(self, log_q_0_T, log_p_0_T):
        weights = self._compute_importance_weights_(log_q_0_T, log_p_0_T)
        forward_KL_per_graph = -jnp.sum(weights * jnp.sum(log_q_0_T, axis=0), axis=-1)
        forward_KL = jnp.mean(forward_KL_per_graph)
        return forward_KL

    def _compute_importance_weights_(self, log_q_0_T, log_p_0_T):
        weights = jax.nn.softmax(jnp.sum(log_p_0_T - log_q_0_T, axis=0), axis=-1)
        return weights

    def _apply_CE(self):
        pass

    def train_step(self, params, opt_state, graph_batch, T, key):

        key, subkey = jax.random.split(key)
        batched_key = jax.random.split(subkey, num=len(jax.devices()))

        (loss, (log_dict, _)), params, opt_state = self.pmap_loss_backward_step(
            params, opt_state, graph_batch, T, batched_key
        )
        return params, opt_state, loss, (log_dict, graph_batch, key)


    def evaluation_step(self, params, graph_batch, T, batched_key, mode="eval", key=None, n_sampling_rounds=None, sampling_temp=None, sampling_mode = "temps", epoch = None, epochs = None):
        start_forw_pass_time = time.time()
        loss, (log_dict, _) = self.pmap_sample(params, graph_batch, T, batched_key)
        end_forw_pass_time = time.time()

        log_dict["time"] = {}
        log_dict["time"]["forward_pass"] = end_forw_pass_time - start_forw_pass_time

        
        log_dict["time"]["CE"] = 0. # can deleted?

        return loss, (log_dict, _)

@partial(jax.jit, static_argnums=())
def repeat_along_nodes(nodes, n_node, target_per_graph):
    total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
    target_per_node = jnp.repeat(target_per_graph, n_node, axis=0,
                                          total_repeat_length=total_nodes)

    return target_per_node
