import jax
import jraph
from flax import struct
from jax import numpy as jnp


@struct.dataclass
class GraphWithMeta:
	graph: jraph.GraphsTuple
	meta: dict = struct.field(pytree_node=False)
	_state: object = struct.field(pytree_node=False, default=None, repr=False)

	_GRAPH_ATTRS = ("nodes", "edges", "senders", "receivers", "globals", "n_node", "n_edge")

	def __getattr__(self, name):
		"""Delegate only GraphsTuple field lookups to the underlying graph."""
		if name in self._GRAPH_ATTRS:
			try:
				graph = object.__getattribute__(self, "graph")
			except AttributeError as exc:  # graph not yet set during unpickling
				raise AttributeError(name) from exc
			return getattr(graph, name)
		try:
			return object.__getattribute__(self, name)
		except AttributeError as exc:
			raise AttributeError(f"{type(self).__name__} has no attribute {name}") from exc


	def get_graph_info(self):
		first_graph = self
		nodes = first_graph.nodes
		n_node = first_graph.n_node
		n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
		graph_idx = jnp.arange(n_graph)
		total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
		node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)
		return node_graph_idx, n_graph, n_node
	

	def __dir__(self):
		"""Expose graph attributes in auto-completion and introspection."""
		return sorted(set(super().__dir__()) | set(self._GRAPH_ATTRS))
	
	def embed_nodes(self, diff_model, X_t, energy_per_node, t_idx_per_node):
		raise NotImplementedError("GraphWithMeta does not implement embed_nodes; use a specific problem subclass.")
	
	def get_prior_logits(self, shape, soft=False):
		raise NotImplementedError("GraphWithMeta does not implement get_prior_logits; use a specific problem subclass.")

	def get_mask(self, shape):
		raise NotImplementedError("GraphWithMeta does not implement get_mask; use a specific problem subclass.")

	def masked_logits_from_scores(self, scores):
		raise NotImplementedError("GraphWithMeta does not implement masked_logits_from_scores; use a specific problem subclass.")
	
	def values_from_embeddings(self, node_embeddings, spin_logits, diff_model):
		print("Warning: using mocked 0-value function in GraphWithMeta.")
		first_graph = self
		n_node = first_graph.n_node
		n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
		return jnp.zeros(n_graph)
	
	def sample_from_logits(self, logits, key):
		print("Warning: using default implementation of sample_from_logits in GraphWithMeta.")
		X = jax.random.categorical(key=key,
										logits=logits,
										axis=-1,
										shape=logits.shape[:-1])[..., None]
		one_hot = jax.nn.one_hot(X[..., 0], num_classes=self.meta["cabinets"])
		return X.astype(jnp.int32), one_hot
	
	def compute_solution_prob_stats(self, spin_logits_next):
		raise NotImplementedError("GraphWithMeta does not implement compute_solution_prob_stats; use a specific problem subclass.")

	def calc_mean_prob(self, spin_log_probs):
		raise NotImplementedError("GraphWithMeta does not implement calc_mean_prob; use a specific problem subclass.")
	


