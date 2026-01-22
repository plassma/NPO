import jax.numpy as jnp
import numpy as np
import jraph
import igraph


def from_igraph_to_jgraph(igraph, zero_edges = True, double_edges = True, _np = np):
    num_vertices = igraph.vcount()
    edge_arr = _np.array(igraph.get_edgelist())
    if(double_edges):
        print("ATTENTION: edges will be dublicated in this method!")
        if(igraph.ecount() > 0):
            undir_receivers = edge_arr[:, 0]
            undir_senders = edge_arr[:, 1]
            receivers = _np.concatenate([undir_receivers[:, np.newaxis], undir_senders[:, np.newaxis]], axis=-1)
            receivers = _np.ravel(receivers)
            senders = _np.concatenate([undir_senders[:, np.newaxis], undir_receivers[:, np.newaxis]], axis=-1)
            senders = _np.ravel(senders)
            edges =  _np.ones((senders.shape[0], 1))
        else:
            receivers = _np.ones((0,), dtype = np.int32)
            senders = _np.ones((0,), dtype = np.int32)
            edges =  _np.ones((0, 1))

        if (not zero_edges):
            edge_weights = igraph.es["weight"]
            edges = _np.concatenate([edge_weights, edge_weights], axis=0)
    else:
        if(igraph.ecount() > 0):
            senders = edge_arr[:, 0]
            receivers = edge_arr[:, 1]
            edges =  _np.ones((senders.shape[0], 1))
        else:
            receivers = _np.ones((0,), dtype = np.int32)
            senders = _np.ones((0,), dtype = np.int32)
            edges =  _np.ones((0, 1))

        if (not zero_edges):
            edge_weights = igraph.es["weight"]
            edges = _np.array(edge_weights)

    nodes = _np.zeros((num_vertices, 1))
    globals = _np.array([num_vertices])
    n_node = _np.array([num_vertices])
    n_edge = _np.array([receivers.shape[0]])

    jgraph = jraph.GraphsTuple(senders = senders, receivers = receivers, edges = edges, nodes = nodes, n_edge = n_edge , n_node = n_node, globals = globals )
    return jgraph