import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

from train import TrainMeanField


def _dataset_paths_exist(dataset_name, seed, problem_name):
    base_path = os.path.dirname(os.getcwd()) + "/DIffUCO/DatasetCreator/loadGraphDatasets/DatasetSolutions/"
    select_data_name = "MaxCl" if problem_name == "MaxClv2" else problem_name
    for mode in ("train", "val"):
        path = os.path.join(
            base_path,
            "no_norm",
            dataset_name,
            mode,
            str(seed),
            select_data_name,
            "indexed",
            "idx_0_solutions.pickle",
        )
        if not os.path.exists(path):
            return False, path
    return True, ""


def _smoke_config():
    config = {
        "save_checkpoints": False,
        "load_wandb_id": None,
        "load_only_params": False,
        "use_sample": 0,
        "n_graphs": 1,
        "mode": "Diffusion",
        "dataset_name": "HCP_dummy",
        "problem_name": "HCP",
        "jit": True,
        "wandb": True,
        "dataloader_num_workers": 0,
        "seed": 123,
        "lr": 1e-4,
        "min_lr": 2.5e-5,
        "batch_size": 1,
        "N_basis_states": 1000,
        "relaxed": True,
        "T_max": 0.005,
        "T_target": 0.006,
        "N_warmup": 0,
        "N_anneal": 300,
        "N_equil": 0,
        "stop_epochs": 50,
        "n_features_list_prob": [64, 2],
        "n_features_list_nodes": [64, 64],
        "n_features_list_edges": [10],
        "n_features_list_messages": [64, 64],
        "n_features_list_encode": [30],
        "n_features_list_decode": [64],
        "message_passing_weight_tied": False,
        "linear_message_passing": True,
        "edge_updates": True,
        "n_diffusion_steps": 4,
        "beta_factor": 0.1,
        "noise_potential": "annealed_obj",
        "time_conditioning": True,
        "piecewise_linear_anneal_schedule": [(0, 0.0015), (2500, 0.005), (5000, 0.006)],
        "project_name": "smoke",
        "mean_aggr": False,
        "grad_clip": True,
        "messeage_concat": False,
        "graph_mode": "normal",
        "loss_alpha": 0.0,
        "train_mode": "GRPO",
        "inner_loop_steps": 2,
        "minib_diff_steps": 4,
        "minib_basis_states": 25,
        "graph_norm": False,
        "proj_method": "None",
        "diff_schedule": "DiffUCO",
        "mov_average": 0.05,
        "sampling_temp": 1.4,
        "n_sampling_rounds": 2,
        "n_test_basis_states": 5,
        "bfloat16": False,
        "AnnealSchedule": "linear",
        "powerlaw_exponent": 4,
        "time_encoding": "learned",
        "lr_schedule": "cosine",
        "TD_k": 3,
        "clip_value": 0.2,
        "value_weighting": 0.65,
        "embedding_dim": 32,
        "node_emb_type": "learned",
        "augment_rooms": False,
        "node_transformer_num_layers": 4,
        "node_transformer_num_heads": 4,
        "node_transformer_dropout_rate": 0.0,
        "transformer_type": "linear",
        "ownership_weight": 1.0,
        "energy_weights": {
            "energy_ownerships": 1.0,
            "energy_things_per_cabinet": 1.0,
            "energy_cabinets_per_room": 1.0,
            "energy_order_violations": 3.0,
            "energy_order_violations_cabinets": 0.0,
            "energy_order_violations_rooms": 0.0,
            "exp_cabinets_things_rooms": 2.0,
            "order_severity": False,
        },
        "load_step": -1,
    }
    config["n_bernoulli_features"] = [10, 20, 30, 100][config["use_sample"]]
    return config


def test_smoke_solution_prob_mean():
    config = _smoke_config()
    exists, missing_path = _dataset_paths_exist(
        config["dataset_name"], config["seed"], config["problem_name"]
    )
    if not exists:
        pytest.skip(f"Missing dataset file: {missing_path}")

    trainer = TrainMeanField(config)
    metrics = trainer.train(
        return_metrics=True,
    )
    assert metrics is not None
    assert "eval/solution_prob_mean" in metrics
    assert metrics["eval/solution_prob_mean"] > -1
