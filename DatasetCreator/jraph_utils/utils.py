import numpy as np
import jraph


def from_igraph_to_jgraph(igraph_graph, zero_edges=True, double_edges=True, _np=np):
    num_vertices = igraph_graph.vcount()

    if igraph_graph.ecount() == 0:
        senders = _np.zeros((0,), dtype=np.int32)
        receivers = _np.zeros((0,), dtype=np.int32)
    else:
        edge_arr = _np.asarray(igraph_graph.get_edgelist(), dtype=np.int32)
        if double_edges:
            senders = _np.concatenate([edge_arr[:, 1], edge_arr[:, 0]], axis=0)
            receivers = _np.concatenate([edge_arr[:, 0], edge_arr[:, 1]], axis=0)
        else:
            senders = edge_arr[:, 0]
            receivers = edge_arr[:, 1]

    if zero_edges:
        edges = _np.ones((senders.shape[0], 1))
    else:
        edge_weights = _np.asarray(igraph_graph.es["weight"])
        if double_edges:
            edge_weights = _np.concatenate([edge_weights, edge_weights], axis=0)
        edges = edge_weights

    nodes = _np.zeros((num_vertices, 1))
    globals = _np.array([num_vertices])
    n_node = _np.array([num_vertices])
    n_edge = _np.array([receivers.shape[0]])

    return jraph.GraphsTuple(
        senders=senders,
        receivers=receivers,
        edges=edges,
        nodes=nodes,
        n_edge=n_edge,
        n_node=n_node,
        globals=globals,
    )


__all__ = ["from_igraph_to_jgraph"]
