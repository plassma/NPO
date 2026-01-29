import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parent.parent
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from Data.LoadGraphDataset import SolutionDatasetLoader
from EnergyFunctions import CountdownEnergyClass
from jraph_utils import pmap_graph_list_better

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    jax.config.update("jax_disable_jit", True)

    base_config = {
        "n_bernoulli_features": 10,
        "n_diffusion_steps": 4,
        "N_basis_states": 10,
        "use_sample": 0,
    }

    energy_fn = CountdownEnergyClass(base_config)

    dataset_loader = SolutionDatasetLoader(
        config=base_config,
        dataset="Countdown_small",
        problem="Countdown",
        batch_size=1,
        relaxed=False,
        seed=123,
        mode="train",
    )

    dataloader_train, dataloader_test, dataloader_val, _ = dataset_loader.dataloaders()
    dataloader = {"train": dataloader_train, "val": dataloader_val, "test": dataloader_test}["train"]
    if dataloader is None:
        raise RuntimeError("No dataloader initialized for mode=train.")


    batch = next(x for i,x in enumerate(iter(dataloader)) if i==0)
    graph_with_meta = batch["input_graph"][0]

    padded_graph = pmap_graph_list_better(graph_with_meta, None)
    padded_graph = jax.tree_util.tree_map(lambda leaf: jnp.asarray(leaf[0]) if hasattr(leaf, "shape") else leaf, padded_graph)

    key = jax.random.PRNGKey(0)
    logits = graph_with_meta.get_prior_logits((padded_graph.nodes.shape[0], 1, 1))
    raw_sample, one_hot = padded_graph.sample_from_logits(logits, key)

    print(raw_sample)

    node_gr_idx, _, _ = padded_graph.get_graph_info()

    prior_energy = energy_fn.calculate_Energy(padded_graph, raw_sample, node_gr_idx)[0]

    solution_energy = energy_fn.calculate_Energy(padded_graph, padded_graph.globals["solution_nodes"].squeeze(), node_gr_idx)[0]

    assert (solution_energy == 0).all()


if __name__ == "__main__":
    main()