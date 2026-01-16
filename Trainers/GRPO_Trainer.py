import jax
import jax.numpy as jnp
from functools import partial

from .PPO_Trainer import PPO


class GRPO(PPO):
    def __init__(self, config, EnergyClass, NoiseClass, model):
        super(GRPO, self).__init__(config, EnergyClass, NoiseClass, model)
        self.c1 = 0.0

    def _calculate_advantages(self, log_dict):
        rewards = log_dict["RL"]["rewards"]
        returns = jnp.sum(rewards, axis=0)
        mean_returns = jnp.mean(returns, axis=-1, keepdims=True)
        std_returns = jnp.std(returns, axis=-1, keepdims=True) + 1e-10
        group_advantages = (returns - mean_returns) / std_returns
        log_dict["RL"]["advantages"] = jnp.broadcast_to(group_advantages, rewards.shape)
        return log_dict

    def _update_policy(self, params, opt_state, graphs, RL_buffer, key):
        log_dict = {"Losses": {"overall_loss": [], "actor_loss": [], "max_ratios": [], "min_ratios": [], "mean_ratios": [], "perc_clipped": []}}

        for i in range(self.inner_loop_steps):
            perm_diff_array, perm_state_array, key = self._shuffle_index_array(key)
            split_diff_array_list = self._split_arrays(perm_diff_array, self.n_diff_batches, -1)
            split_state_array_list = self._split_arrays(perm_state_array, self.n_diff_batches, -1)

            for split_diff_arr, split_state_arr in zip(split_diff_array_list, split_state_array_list):
                split_split_diff_arr_list = self._split_arrays(split_diff_arr, self.n_state_batches, -2)
                split_split_state_arr_list = self._split_arrays(split_state_arr, self.n_state_batches, -2)
                for split_split_diff_arr, split_split_state_arr in zip(split_split_diff_arr_list, split_split_state_arr_list):
                    (loss, (loss_dict, key)), params, opt_state = self.loop_inner(
                        params, opt_state, graphs, RL_buffer, key, split_split_diff_arr, split_split_state_arr
                    )

                    for dict_key in log_dict["Losses"].keys():
                        log_dict["Losses"][dict_key].append(loss_dict[dict_key])

        return (loss, (log_dict, key)), params, opt_state

    @partial(jax.jit, static_argnums=(0,))
    def PPO_loss(self, params, jraph_graph_list, batch_dict, key):
        Sb_Hb_Nb_A_k = batch_dict["advantages"]
        Sb_Hb_Nb_X_prev = batch_dict["states"]
        Sb_Hb_Nb_X_next = batch_dict["actions"]

        Sb_Nb_t_idx_per_node = batch_dict["time_index_per_node"]
        Sb_Hb_Nb_state_log_probs = batch_dict["policies"]

        key, subkey = jax.random.split(key)
        batched_key = jax.random.split(subkey, num=Sb_Hb_Nb_A_k.shape[0])

        node_gr_idx, n_graph, total_num_nodes = self._compute_aggr_utils(jraph_graph_list["graphs"][0])
        energy_per_node = self.vmapped_relaxed_energy(
            jraph_graph_list["graphs"][0],
            Sb_Hb_Nb_X_prev.swapaxes(0, 1).astype(jnp.int32),
            node_gr_idx,
        )[2]
        out_dict, _ = self.vmapped_calc_log_q(
            params,
            jraph_graph_list,
            Sb_Hb_Nb_X_prev,
            energy_per_node.T,
            Sb_Hb_Nb_X_next,
            Sb_Nb_t_idx_per_node,
            batched_key,
        )

        state_log_probs = out_dict["state_log_probs"]
        ratios = jnp.exp(state_log_probs - Sb_Hb_Nb_state_log_probs)

        surr1 = ratios * Sb_Hb_Nb_A_k
        surr2 = jnp.clip(ratios, 1 - self.clip_value, 1 + self.clip_value) * Sb_Hb_Nb_A_k

        actor_loss = jnp.mean((-jnp.minimum(surr1, surr2)[:, :-1]))
        overall_loss = actor_loss

        max_ratios = jnp.max(ratios)
        min_ratios = jnp.min(ratios)
        mean_ratios = jnp.mean(ratios)
        clip_fraction = jnp.mean(1 * (ratios < 1 - self.clip_value) + 1 * (ratios > 1 + self.clip_value))

        loss_dict = {
            "actor_loss": actor_loss,
            "overall_loss": overall_loss,
            "max_ratios": max_ratios,
            "min_ratios": min_ratios,
            "mean_ratios": mean_ratios,
            "perc_clipped": clip_fraction,
        }
        return overall_loss, (loss_dict, key)
