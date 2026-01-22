import jraph
import jax.numpy as jnp
import numpy as np
import itertools
import jax
import time
import jax.tree_util as tree
from GraphWithMeta import GraphWithMeta

def global_graph_aggr(feature, node_graph_idx, n_graph):
		aggr_feature = jax.ops.segment_sum(feature, node_graph_idx, n_graph)
		return aggr_feature

def _ensure_list(graphs):
    if isinstance(graphs, (list, tuple)):
        return list(graphs)
    return [graphs]


def _repeat_graphs_for_devices(graph_list, n_devices):
    graph_list = _ensure_list(graph_list)
    if len(graph_list) == 0:
        raise ValueError("graph_list must contain at least one element.")

    target_len = int(np.ceil(len(graph_list) / n_devices) * n_devices)
    target_len = max(target_len, n_devices)
    if len(graph_list) < target_len:
        graph_list = list(itertools.islice(itertools.cycle(graph_list), target_len))
    n_graphs_per_device = target_len // n_devices
    return graph_list, n_graphs_per_device


def _meta_value_equal(lhs, rhs):
    lhs_arr = np.asarray(lhs)
    rhs_arr = np.asarray(rhs)
    if lhs_arr.shape == () and rhs_arr.shape == ():
        return bool(lhs_arr == rhs_arr)
    return np.array_equal(lhs_arr, rhs_arr)


def _meta_dict_equal(lhs, rhs):
    if lhs.keys() != rhs.keys():
        return False
    return all(_meta_value_equal(lhs[k], rhs[k]) for k in lhs.keys())


def _merge_meta(meta_list):
    base_meta = dict(meta_list[0])
    if not all(_meta_dict_equal(base_meta, meta) for meta in meta_list[1:]):
        base_meta = dict(meta_list[0])
    base_meta["n_graphs"] = len(meta_list)
    base_meta.update({f"{k}_concat": [meta[k] for meta in meta_list] for k in meta_list[0].keys()})
    return base_meta


def pad_with_graphs(graph: jraph.GraphsTuple,
                    n_node: int,
                    n_edge: int,
                    n_graph: int = 2) -> jraph.GraphsTuple:
    """Pads a ``GraphsTuple`` to size by adding computation preserving graphs.

    The ``GraphsTuple`` is padded by first adding a dummy graph which contains the
    padding nodes and edges, and then empty graphs without nodes or edges.

    The empty graphs and the dummy graph do not interfer with the graphnet
    calculations on the original graph, and so are computation preserving.

    The padding graph requires at least one node and one graph.

    This function does not support jax.jit, because the shape of the output
    is data-dependent.

    Args:
    graph: ``GraphsTuple`` padded with dummy graph and empty graphs.
    n_node: the number of nodes in the padded ``GraphsTuple``.
    n_edge: the number of edges in the padded ``GraphsTuple``.
    n_graph: the number of graphs in the padded ``GraphsTuple``. Default is 2,
      which is the lowest possible value, because we always have at least one
      graph in the original ``GraphsTuple`` and we need one dummy graph for the
      padding.

    Raises:
    ValueError: if the passed ``n_graph`` is smaller than 2.
    RuntimeError: if the given ``GraphsTuple`` is too large for the given
      padding.

    Returns:
    A padded ``GraphsTuple``.
    """
    if n_graph < 2:
        raise ValueError(f'n_graph is {n_graph}, which is smaller than minimum value of 2.')
    pad_n_node = int(n_node - np.sum(graph.n_node))
    pad_n_edge = int(n_edge - np.sum(graph.n_edge))
    pad_n_graph = int(n_graph - graph.n_node.shape[0])
    if pad_n_node <= 0 or pad_n_edge < 0 or pad_n_graph <= 0:
        raise RuntimeError(
            'Given graph is too large for the given padding. difference: '
            f'n_node {pad_n_node}, n_edge {pad_n_edge}, n_graph {pad_n_graph}')

    pad_n_empty_graph = pad_n_graph - 1

    tree_nodes_pad = (
        lambda leaf: np.zeros((pad_n_node,) + leaf.shape[1:], dtype=leaf.dtype))
    tree_edges_pad = (
        lambda leaf: np.zeros((pad_n_edge,) + leaf.shape[1:], dtype=leaf.dtype))
    tree_globs_pad = (
        lambda leaf: np.zeros((pad_n_graph,) + leaf.shape[1:], dtype=leaf.dtype))

    padding_graph = jraph.GraphsTuple(
        n_node=np.concatenate(
            [np.array([pad_n_node], dtype=np.int32),
             np.zeros(pad_n_empty_graph, dtype=np.int32)]),
        n_edge=np.concatenate(
            [np.array([pad_n_edge], dtype=np.int32),
             np.zeros(pad_n_empty_graph, dtype=np.int32)]),
        nodes=tree.tree_map(tree_nodes_pad, graph.nodes),
        edges=tree.tree_map(tree_edges_pad, graph.edges),
        globals=tree.tree_map(tree_globs_pad, graph.globals),
        senders=np.zeros(pad_n_edge, dtype=np.int32),
        receivers=np.zeros(pad_n_edge, dtype=np.int32),
    )
    return jraph.batch_np([graph, padding_graph])

def __nearest_bigger_power_of_k(x: int, k: float) -> int:
    """Computes the nearest power of two greater than x for padding."""
    if x == 0:
        return 0
    exponent = np.log(x) / np.log(k)
    return int(k**(int(exponent) + 1))



def calc_pad_number_from_statistics(graphs_tuple, dataset_statistics_dict, np_ = jnp):

    min_edges = dataset_statistics_dict["min_edges"]
    max_edges = dataset_statistics_dict["max_edges"]

    min_nodes = dataset_statistics_dict["min_nodes"]
    max_nodes = dataset_statistics_dict["max_nodes"]

    grid_num = dataset_statistics_dict["grid_num"]

    pad_nodes_to = _nearest_number_of_min_max(min_nodes, max_nodes, np_.sum(graphs_tuple.n_node), grid_num, len(graphs_tuple.n_node)) + 1
    pad_edges_to = _nearest_number_of_min_max(min_edges, max_edges, np_.sum(graphs_tuple.n_edge), grid_num, len(graphs_tuple.n_node)) + 1
    return pad_nodes_to, pad_edges_to

def _nearest_number_of_min_max(min_value, max_value, graph_value, grid_num, n_graphs):
    if(min_value == max_value):
        return max_value
    elif(n_graphs == 1):
        return max_value
    else:
        grid_candidates = np.linspace(min_value, max_value, grid_num, endpoint=True)
        return int(find_smallest_greater_value(grid_candidates, graph_value))

def find_smallest_greater_value(grid_candidates, value):
    mask = grid_candidates >= value
    if np.any(mask):
        return np.min(grid_candidates[mask])
    else:
        return None

def pad_graphs_to_same_size_from_statistics(graphs_list, dataset_statistics_dict, pad_func = pad_with_graphs):
    max_pad_nodes_to = 1
    max_pad_edges_to = 1
    for graph in graphs_list:
        pad_nodes_to, pad_edges_to = calc_pad_number_from_statistics(graph, dataset_statistics_dict)
        max_pad_nodes_to = max([max_pad_nodes_to, pad_nodes_to])
        max_pad_edges_to = max([max_pad_edges_to, pad_edges_to])

    max_pad_nodes_to = np.max([max_pad_nodes_to, 1])
    # print("here")
    # for graph in graphs_list:
    #     print(len(graphs_list), )
    #     print((graph.nodes.shape, graph.edges.shape, pad_nodes_to, pad_edges_to, graph.n_node.shape[0] + 1))
    #     print(jraph.pad_with_graphs(graph, pad_nodes_to, pad_edges_to, graph.n_node.shape[0] + 1))
    #     print("finished")
    padded_graph_list = [pad_func(graph, max_pad_nodes_to, max_pad_edges_to, graph.n_node.shape[0] + 1) for graph in graphs_list]
    return padded_graph_list, max_pad_nodes_to, max_pad_edges_to



def device_batch(graph_generator, np_ = np):
    """Batches a set of graphs the size of the number of devices."""
    num_devices = jax.local_device_count()
    batch = []
    for idx, graph in enumerate(graph_generator):
        if idx % num_devices == num_devices - 1:
            batch.append(graph)
            yield jax.tree_util.tree_map(lambda *x: np_.stack(x, axis=0), *batch) ### TODO text wheter numpy or jnp is better here
            batch = []
        else:
            batch.append(graph)


def pmap_graph_list_better(maybe_meta_graph_list, dataset_statistics_dict, pad_func = pad_with_graphs, return_size = False):
    is_meta = False
    n_devices = jax.local_device_count()
    maybe_meta_graph_list = _ensure_list(maybe_meta_graph_list)
    if isinstance(maybe_meta_graph_list[0], GraphWithMeta):
        is_meta = True

    maybe_meta_graph_list, n_graphs_per_device = _repeat_graphs_for_devices(maybe_meta_graph_list, n_devices)

    if is_meta:
        meta_list = [el.meta for el in maybe_meta_graph_list]
        jraph_graph_list = [el.graph for el in maybe_meta_graph_list]
    else:
        jraph_graph_list = maybe_meta_graph_list
    
    device_batched_graphs = [jraph.batch_np(jraph_graph_list[idx * n_graphs_per_device: (idx + 1) * n_graphs_per_device])
                             for idx in range(n_devices)] ### TODO move this to collate function


    padded_graph_list, max_pad_nodes_to, max_pad_edges_to = pad_graphs_to_same_size_from_statistics(device_batched_graphs, dataset_statistics_dict, pad_func = pad_func)
    device_batched_graphs = next(device_batch(padded_graph_list))
    # print("make list", step2-step1)
    # print("pad graphs", step3-step2)
    # print("next generator", step4-step3)
    if is_meta:
        meta = _merge_meta(meta_list)
        device_batched_graphs = GraphWithMeta(graph=device_batched_graphs, meta=meta)
    if(return_size):
        return device_batched_graphs, max_pad_nodes_to, max_pad_edges_to
    else:
        return device_batched_graphs
