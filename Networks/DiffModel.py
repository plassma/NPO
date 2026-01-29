from functools import partial

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from jraph_utils import global_graph_aggr
from Networks.Modules.MLPModules.MLPs import ValueMLP
from Networks.Modules.Transformer.LinearTransformerEncoderStack import (
	LinearTransformerEncoderStack,
)
from Networks.Modules.Transformer.TransformerEncoderStack import TransformerEncoderStack
from Networks.Modules.xLSTM.mlstm import mLSTMStack


class DiffModel(nn.Module):
	"""
	Policy Network
	"""
	n_features_list_prob: np.ndarray

	n_features_list_nodes: np.ndarray
	n_features_list_edges: np.ndarray
	n_features_list_messages: np.ndarray

	n_features_list_encode: np.ndarray
	n_features_list_decode: np.ndarray

	n_diffusion_steps: int
	edge_updates: bool
	problem_type: str

	time_encoding: str
	n_diff_steps: int
	embedding_dim: int = 32
	message_passing_weight_tied: bool = True
	linear_message_passing: bool = True
	n_bernoulli_features: int = 1
	mean_aggr: bool = False
	EncoderModel: str = "normal"
	dim_energy_per_node: int = 1
	train_mode: str = "REINFORCE"
	graph_norm: bool = False
	bfloat16: bool = False
	dataset_name: str = "None"
	augment_rooms: bool = False
	node_emb_type: str = "neighbor"
	node_transformer_num_layers: int = 4
	node_transformer_num_heads: int = 4
	node_transformer_dropout_rate: float = 0.0
	transformer_type: str = "linear"  # "linear" or "standard"
	value_pooling: str = "graph_attention"  # "neighbor_attention", "node_sum", "graph_attention", # gpu0: GAT smaller value gate, GPU1: deeper value mlp



	def setup(self):
		if not self.bfloat16 :
			dtype = jnp.float32
		else:
			dtype = jnp.bfloat16

		# Countdown specific stuff

		self.number_encoder = nn.Sequential([nn.Dense(features=self.embedding_dim, dtype=dtype)])
		self.symbol_encoder = nn.Embed(num_embeddings=50, features=self.embedding_dim, dtype=dtype)

		self.category_head = nn.Sequential([
			nn.Dense(features=self.embedding_dim * 4, dtype=dtype),
			nn.gelu,
			nn.Dense(features= self.n_bernoulli_features, dtype=dtype),
		])

		#general stuff
		self.feature_proj = nn.Sequential([
			nn.Dense(features=self.embedding_dim * 4, dtype=dtype),
			nn.gelu,
			nn.Dense(features=self.embedding_dim, dtype=dtype),
			nn.LayerNorm(dtype=dtype)
		])

		self.node_transformer = TransformerEncoderStack(
	            num_layers=self.node_transformer_num_layers,
	            embed_dim=self.embedding_dim,
	            mlp_dim=self.embedding_dim * 4,
	            num_heads=self.node_transformer_num_heads,
	            dropout_rate=self.node_transformer_dropout_rate,
	            dtype=dtype,
	        )
		
		self.linear_node_transformer = LinearTransformerEncoderStack(
	            num_layers=self.node_transformer_num_layers,
	            embed_dim=self.embedding_dim,
	            mlp_dim=self.embedding_dim * 4,
	            num_heads=self.node_transformer_num_heads,
	            dropout_rate=self.node_transformer_dropout_rate,
	            dtype=dtype,
	        )
		
		self.mLSTMStack = mLSTMStack(
			num_layers=self.node_transformer_num_layers,
			embed_dim=self.embedding_dim,
			num_heads=self.node_transformer_num_heads,
			causal=False,
		)
		
		# HCP specific stuff
		self.W_k = nn.Sequential([nn.Dense(features=self.embedding_dim * 2, dtype=dtype), nn.gelu, nn.Dense(features=self.embedding_dim, dtype=dtype)])
		self.W_q = nn.Sequential([nn.Dense(features=self.embedding_dim * 2, dtype=dtype), nn.gelu, nn.Dense(features=self.embedding_dim, dtype=dtype)])
		self.W_v = nn.Sequential([nn.Dense(features=self.embedding_dim * 2, dtype=dtype), nn.gelu, nn.Dense(features=self.embedding_dim, dtype=dtype)])
		self.value_gate = nn.Sequential([nn.Dense(features=self.embedding_dim // 2, dtype=dtype), nn.gelu, nn.Dense(features=1, dtype=dtype)])
		
		self.value_mlp = ValueMLP(n_features_list=[self.embedding_dim * 2, self.embedding_dim * 2,1], dtype=dtype)

		self.time_step_emb = nn.Embed(num_embeddings=self.n_diffusion_steps, features=8, dtype=dtype)
		self.node_type_emb = nn.Embed(num_embeddings=5, features=8, dtype=dtype)
		self.nth_of_type_emb = nn.Embed(num_embeddings=500, features=32, dtype=dtype)
		self.connection_emb = [
			nn.Embed(num_embeddings=500, features=self.embedding_dim, dtype=dtype), # rooms -> persons
			nn.Embed(num_embeddings=50, features=self.embedding_dim, dtype=dtype), # cabinets -> rooms
			nn.Embed(num_embeddings=100, features=self.embedding_dim, dtype=dtype), # things -> cabinets
			nn.Embed(num_embeddings=50, features=self.embedding_dim, dtype=dtype), # ownership -> persons
			nn.Embed(num_embeddings=1, features=self.embedding_dim, dtype=dtype)  # persons (dummy)
		]


	@partial(flax.linen.jit, static_argnums=(0,), static_argnames=("deterministic",))
	def __call__(self, graph_batch, X_prev, energy_per_node, t_idx_per_node, key, *, deterministic: bool = True):
		X_prev = X_prev.astype(jnp.int32)
		X_prev_emb = graph_batch.embed_nodes(self, X_prev, energy_per_node, t_idx_per_node)
		X_prev_emb = jnp.concatenate([X_prev_emb], axis = -1)

		if self.transformer_type == "linear":
			embeddings = self.linear_node_transformer(self.feature_proj(X_prev_emb), deterministic=deterministic)  # deterministic=deterministic
		elif self.transformer_type == "standard":
			embeddings = self.node_transformer(self.feature_proj(X_prev_emb))  # deterministic=deterministic
		elif self.transformer_type == "mlstm":
			embeddings = self.mLSTMStack(self.feature_proj(X_prev_emb)[None])[0]
		else:
			raise ValueError(f"Unknown transformer_type: {self.transformer_type}")

		if self.problem_type == "Countdown":
			# Countdown specific head
			scores = self.category_head(embeddings)
		else:
			queries = self.W_q(embeddings)
			keys = self.W_k(embeddings)[graph_batch.globals["neighbours_per_node"]]

			scores = jnp.einsum('nd,ncd->nc', queries, keys) / jnp.sqrt(queries.shape[-1]) #score_embeddings

			#scores = self.category_head(embeddings)
		
		spin_logits = graph_batch.masked_logits_from_scores(scores)
		
		Values = graph_batch.values_from_embeddings(embeddings, spin_logits, self)

		out_dict = {
			"spin_logits": spin_logits,
			"Values": Values
		}
		return out_dict, key

	@partial(flax.linen.jit, static_argnums=0, static_argnames=("deterministic",))
	def make_one_step(self,params ,graph_batch, X_prev, energy_per_node, t_idx_per_node, key, deterministic: bool = True, step: int = 0):
		rngs = None
		if not deterministic:
			key, dropout_key = jax.random.split(key)
			rngs = {"dropout": dropout_key}
		
		node_graph_idx, n_graph, n_node = graph_batch.get_graph_info()

		out_dict, key = self.apply(params, graph_batch, X_prev, energy_per_node, t_idx_per_node, key, deterministic=deterministic, rngs=rngs)
		X_next, spin_log_probs, key = self.sample_from_model(out_dict["spin_logits"], graph_batch, key)

		graph_log_prob = jax.lax.stop_gradient(jnp.exp((self.__get_log_prob(spin_log_probs[...,0], node_graph_idx, n_graph)/(n_node))[:-1]))
		out_dict["X_next"] = X_next[..., 0]
		out_dict["spin_log_probs"] = spin_log_probs
		out_dict["state_log_probs"] = self.__get_log_prob(spin_log_probs[...,0], node_graph_idx, n_graph)
		out_dict["graph_log_prob"] = graph_log_prob
		return out_dict, key
	
	
	@partial(flax.linen.jit, static_argnums=0)
	def sample_from_model(self, spin_logits, graph_batch, key):
		key, subkey = jax.random.split(key)
		X_next, one_hot_state = graph_batch.sample_from_logits(spin_logits, key=subkey) # logits: 121, 1, 10, x_next: 121, 1, 1

		spin_log_probs = jnp.sum(spin_logits * one_hot_state, axis=-1)
		return X_next, spin_log_probs, key

	
	@partial(flax.linen.jit, static_argnums=0)
	def calc_log_q(self, params, graph_batch, X_prev, energy_per_node, X_next, t_idx_per_node, key):
		out_dict, key = self.apply(params, graph_batch, X_prev, energy_per_node, t_idx_per_node, key)

		spin_logits = out_dict["spin_logits"]
		node_graph_idx, n_graph, _ = graph_batch.get_graph_info()

		one_hot_state = jax.nn.one_hot(X_next, num_classes=self.n_bernoulli_features)
		spin_log_probs = jnp.sum(spin_logits * one_hot_state, axis=-1)
		X_next_log_prob = self.__get_log_prob(spin_log_probs[...,0], node_graph_idx, n_graph)


		out_dict["state_log_probs"] = X_next_log_prob
		out_dict["spin_log_probs"] = spin_log_probs
		return out_dict, key

	@partial(flax.linen.jit, static_argnums=0)
	def calc_log_q_T(self, j_graph, X_T):
		'''

		:param j_graph:
		:param X_T: shape =  (batched_graph_nodes, n_states, 1)
		:return:
		'''

		shape = X_T.shape
		log_p_uniform = self._get_prior(shape, j_graph, soft=True)

		one_hot_state = jax.nn.one_hot(X_T[..., 0], num_classes=self.n_bernoulli_features)
		log_p_X_T_per_node = jnp.sum(log_p_uniform * one_hot_state, axis=-1)


		nodes = j_graph.nodes
		n_node = j_graph.n_node
		n_graph = j_graph.n_node.shape[0]
		graph_idx = jnp.arange(n_graph)
		total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
		node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)

		log_p_X_T = self.__get_log_prob(log_p_X_T_per_node, node_graph_idx, n_graph)

		return log_p_X_T

	def sample_prior(self, j_graph, N_basis_states, key):
		nodes = j_graph.nodes
		shape = (nodes.shape[0], N_basis_states, 1)
		key, subkey = jax.random.split(key)
		log_p_uniform = self._get_prior(shape, j_graph)

		X_prev, one_hot_state = j_graph.sample_from_logits(log_p_uniform, key=subkey)
		return X_prev, one_hot_state, log_p_uniform, key

	def sample_prior_w_probs(self, j_graph, N_basis_states, key):
		X_T, one_hot_state, log_p_uniform, key = self.sample_prior(j_graph, N_basis_states, key)
		log_p_X_T = self.calc_log_q_T(j_graph, X_T)
		return X_T, log_p_X_T, one_hot_state, log_p_uniform, key


	@partial(flax.linen.jit, static_argnums=(0,1))
	def _get_prior(self, shape, meta_graph, soft=False):
		return meta_graph.get_prior_logits(shape, soft=soft)
	
	def __get_log_prob(self, spin_log_probs, node_graph_idx, n_graph):
			log_probs = global_graph_aggr(spin_log_probs, node_graph_idx, n_graph)
			return log_probs

	
