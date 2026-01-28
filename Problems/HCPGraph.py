
import jax
from flax import struct
from jax import numpy as jnp

from house_config import prior_logits_for_graph
from jraph_utils import global_graph_aggr
from Problems.GraphWithMeta import GraphWithMeta
from utils.utils import segment_softmax, vmap_get_sinusoidal_positional_encoding


@struct.dataclass
class HCPGraph(GraphWithMeta):
	
	def embed_nodes(self, diff_model, X_t, energy_per_node, t_idx_per_node):
		dtype = jnp.bfloat16 if diff_model.bfloat16 else jnp.float32
		t_idx = jnp.squeeze(t_idx_per_node, axis=-1)
		t_idx = jnp.clip(t_idx, 0, diff_model.n_diff_steps - 1)
		
		if(diff_model.time_encoding == "one_hot"):
			T_embed = jax.nn.one_hot(t_idx, num_classes=diff_model.n_diffusion_steps)
		elif (diff_model.time_encoding == "learned"):
			T_embed = diff_model.time_step_emb(t_idx.astype(jnp.int32)).astype(dtype)
		else:
			T_embed = vmap_get_sinusoidal_positional_encoding(t_idx, diff_model.embedding_dim // 4).astype(dtype)

		node_types_emb = diff_model.node_type_emb(self.globals["node_types"])

		nth_of_type_emb = vmap_get_sinusoidal_positional_encoding(self.globals["nth_of_type"], 16).astype(dtype) # self.nth_of_type_emb(jraph_graph.globals["nth_of_type"])

		energy_per_node_emb = vmap_get_sinusoidal_positional_encoding((energy_per_node.squeeze()).astype(jnp.int32), 8).astype(dtype)
		
		X_input = jnp.concatenate([T_embed, node_types_emb, nth_of_type_emb, jnp.zeros_like(energy_per_node_emb)], axis=-1) # we don't need rand_nodes! # energy_per_node_emb

		if diff_model.node_emb_type == "one_hot":
			X_emb = jax.nn.one_hot(X_t[..., 0], num_classes=self.meta["cabinets"])
		elif diff_model.node_emb_type == "neighbor":
			idx = jnp.array(self.globals["neighbours_per_node"])[jnp.arange(X_t.shape[0]), X_t[:, 0].astype(jnp.int32)]
			X_emb = X_input[idx]
		elif diff_model.node_emb_type == "learned":
			X_emb = jnp.zeros((X_t.shape[0], diff_model.embedding_dim + 16), dtype=dtype)
			slices = [slice(self.meta["offset_rooms"], self.meta["offset_rooms"] + self.meta["rooms"]), 
					  slice(self.meta["offset_cabinets"], self.meta["offset_cabinets"] + self.meta["cabinets"]), 
					  slice(self.meta["offset_things_cabinets"], self.meta["offset_things_cabinets"] + self.meta["things"]), 
					  slice(self.meta["offset_things_persons"], self.meta["offset_things_persons"] + self.meta["things"]),
					  slice(self.meta["offset_persons"], self.meta["offset_persons"] + self.meta["persons"])]
			for i in range(len(slices)):
				X_emb = X_emb.at[slices[i]].set(jnp.concatenate([diff_model.connection_emb[i](X_t[slices[i], 0].astype(jnp.int32)), vmap_get_sinusoidal_positional_encoding(X_t[slices[i], 0], 16)], axis=-1))
		else:
			X_emb = vmap_get_sinusoidal_positional_encoding(X_t[..., 0], diff_model.embedding_dim)

		X_emb = X_emb.astype(dtype)
		X_input = jnp.concatenate([X_input, X_emb, ], axis=-1).astype(dtype) # energy_per_node_emb

		
		return X_input
	
	def get_prior_logits(self, shape, soft=False):
		base_logits = prior_logits_for_graph(self, soft=soft)
		target_shape = shape[:-1] + (base_logits.shape[-1],)
		return jnp.broadcast_to(base_logits[:, None, :], target_shape)

	def get_mask(self):
		mask = (jnp.arange(self.meta["cabinets"])[None, :] < self.globals["classes_per_node"][:, None])
		return mask * 1.0
	
	def masked_logits_from_scores(self, scores):
		mask = self.get_mask()
		mask_bool = mask.astype(bool)
		neg_inf = jnp.array(jnp.finfo(scores.dtype).min, dtype=scores.dtype)
		masked_scores = jnp.where(mask_bool, scores, neg_inf)
		all_masked = jnp.all(~mask_bool, axis=-1, keepdims=True)
		safe_scores = jnp.where(all_masked, jnp.zeros_like(masked_scores), masked_scores)
		spin_logits = jax.nn.log_softmax(safe_scores, axis=-1)
		spin_logits = jnp.where(mask_bool, spin_logits, neg_inf)
		spin_logits = jnp.where(all_masked, neg_inf, spin_logits)[:, None]
		
		#fix ownership logits constant
		start_ownerships = self.meta["offset_things_persons"]
		end_ownerships = start_ownerships + self.meta["things"]
		spin_logits = spin_logits.at[start_ownerships:end_ownerships].set(neg_inf)
		rows = jnp.arange(start_ownerships, end_ownerships)
		cols = self.globals["owners_of_things"][:-1]
		spin_logits = spin_logits.at[rows, 0, cols].set(0.0)

		return spin_logits
	
	def values_from_embeddings(self, node_embeddings, spin_logits, diff_model):
		node_graph_idx, n_graph, n_node = self.get_graph_info()
		
		if diff_model.value_pooling == "neighbor_attention":
			attn_weights = jnp.exp(spin_logits[:, 0, :])
			attn_weights = attn_weights / jnp.clip(jnp.sum(attn_weights, axis=-1, keepdims=True), a_min=1e-9)
			value_values = diff_model.W_v(node_embeddings)[self.globals["neighbours_per_node"]]
			attn_out = jnp.einsum('nc,ncd->nd', attn_weights, value_values)
			value_emb = global_graph_aggr(attn_out[:, None], node_graph_idx, n_graph) / jnp.sqrt(n_node[..., None, None])
		elif diff_model.value_pooling == "node_sum":
			value_values = diff_model.W_v(node_embeddings)
			value_emb = global_graph_aggr(value_values[:, None], node_graph_idx, n_graph) / jnp.sqrt(n_node[..., None, None])
		elif diff_model.value_pooling == "graph_attention":
			value_values = diff_model.W_v(node_embeddings)
			value_logits = diff_model.value_gate(node_embeddings)[..., 0]
			value_weights = segment_softmax(value_logits, node_graph_idx, n_graph)
			weighted_values = value_values * value_weights[:, None]
			value_emb = global_graph_aggr(weighted_values[:, None], node_graph_idx, n_graph) / jnp.sqrt(n_node[..., None, None])
		else:
			raise ValueError(f"Unknown value_pooling: {diff_model.value_pooling}")
		
		Values = diff_model.value_mlp(value_emb)[..., 0, 0]

		return Values