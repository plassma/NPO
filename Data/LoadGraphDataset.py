import os
import random

import jax
from torch.utils.data import Dataset
import pickle
import numpy as np
import igraph
import jraph
import torch
from torch.utils.data import DataLoader
from unipath import Path
import os
import jraph_utils
from GraphWithMeta import GraphWithMeta


class SolutionDatasetLoader:
    def __init__(self, config = {}, dataset="MIS", problem="MIS", batch_size=32, relaxed=False, seed=123, mode = "train"):
        self.dataset_name = dataset
        self.problem_name = problem
        self.batch_size = batch_size
        self.relaxed = relaxed
        self.seed = seed
        self.config = config
        self.mode = mode

        default_workers = max(self.batch_size, 40)
        if self.config.get("dataset_in_memory", True):
            default_workers = 0
        self.num_workers = self.config.get("dataloader_num_workers", default_workers)
        self.prefetch_factor = self.config.get("dataloader_prefetch_factor", 2)
        if self.num_workers > 0:
            self.persistent_workers = self.config.get("dataloader_persistent_workers", True)
        else:
            self.persistent_workers = False

        torch.manual_seed(self.seed)
        self._init_mode()

    def _init_mode(self):
        if(self.mode == "train"):
            self.train_dataset = True
            self.val_dataset = True
            self.test_dataset = False
        elif(self.mode == "val"):
            self.train_dataset = False
            self.val_dataset = True
            self.test_dataset = False
        else:
            self.train_dataset = False
            self.val_dataset = False
            self.test_dataset = True

    def pmap_collate(self, batch):
        #batch_transposed = list(zip(*batch))
        batch_dict = {key: [] for key in batch[0].keys()}

        for el in batch:
            for key in batch_dict.keys():
                batch_dict[key].append(el[key])

        # jraph_graphs = [el["input_graph"] for el in batch]
        # energy_graphs = [el["energy_graph"] for el in batch]
        # gt_normed_energies = [el["energies"] for el in batch]
        # gt_spin_states = [el["gs_bins"] for el in batch]
        # U_net_graphs_dict = [el["U_net_graphs_dict"] for el in batch]
        # #print(gt_spin_states)
        return batch_dict

    def dataloaders(self):


        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        TRAIN_DATASET = self.train_dataset
        TEST_DATASET = self.test_dataset
        VAL_DATASET = self.val_dataset

        dataset_train = SolutionDataset_InMemory(config = self.config, dataset=self.dataset_name, problem=self.problem_name, mode="train", relaxed=self.relaxed, seed=self.seed) if TRAIN_DATASET else None
        dataset_test = SolutionDataset_InMemory(config = self.config, dataset=self.dataset_name, problem=self.problem_name, mode="test", relaxed=self.relaxed, seed=self.seed) if TEST_DATASET else None
        dataset_val = SolutionDataset_InMemory(config = self.config, dataset=self.dataset_name, problem=self.problem_name, mode="val", relaxed=self.relaxed, seed=self.seed) if VAL_DATASET else None

        if(self.mode == "train"):
            mean_energy = dataset_train.val_mean_energy
            std_energy = dataset_train.val_std_energy
        elif(self.mode == "val"):
            mean_energy = dataset_val.val_mean_energy
            std_energy = dataset_val.val_std_energy
        else:
            mean_energy = dataset_test.val_mean_energy
            std_energy = dataset_test.val_std_energy

        collate_function = self.pmap_collate

        def _loader_kwargs(**extra_kwargs):
            loader_kwargs = dict(
                batch_size=self.batch_size,
                collate_fn=collate_function,
                num_workers=self.num_workers,
                worker_init_fn=seed_worker,
                generator=generator
            )
            loader_kwargs.update(extra_kwargs)
            if self.num_workers > 0:
                loader_kwargs["prefetch_factor"] = self.prefetch_factor
                loader_kwargs["persistent_workers"] = self.persistent_workers
            return loader_kwargs

        self.dataset_train = dataset_train
        if TRAIN_DATASET:
            train_kwargs = _loader_kwargs(drop_last=False, shuffle=True)
            self.dataloader_train = DataLoader(self.dataset_train, **train_kwargs)
        else:
            self.dataloader_train = None

        if TEST_DATASET:
            test_kwargs = _loader_kwargs()
            self.dataloader_test = DataLoader(dataset_test, **test_kwargs)
        else:
            self.dataloader_test = None

        if VAL_DATASET:
            val_kwargs = _loader_kwargs()
            self.dataloader_val = DataLoader(dataset_val, **val_kwargs)
        else:
            self.dataloader_val = None
        if(self.dataloader_train != None):
            self._compute_dataset_statistics(mode = "train")
        if(self.dataloader_val != None):
            self._compute_dataset_statistics(mode = "val")
        if(self.dataloader_test != None):
            self._compute_dataset_statistics(mode = "test")
        return self.dataloader_train, self.dataloader_test, self.dataloader_val, (mean_energy, std_energy)

    def _compute_dataset_statistics(self, mode = "train"):

        if(mode == "train"):
            current_dataloader = self.dataloader_train
        elif(mode == "val"):
            current_dataloader = self.dataloader_val
        else:
            current_dataloader = self.dataloader_test

        statistics_dict = {}
        statistics_dict["input_graph"] = {"n_edges": [], "n_nodes": []}
        statistics_dict["energy_graph"] = {"n_edges": [], "n_nodes": []}

        for batch_dict in current_dataloader:
            input_graph = batch_dict["input_graph"]
            energy_graph = batch_dict["energy_graph"]
            energy_graph_nodes = [int(el.graph.n_node[0]) for el in energy_graph]
            input_graph_nodes = [int(el.graph.n_node[0]) for el in input_graph]
            energy_graph_edges = [int(el.graph.n_edge[0]) for el in energy_graph]
            input_graph_edges = [int(el.graph.n_edge[0]) for el in input_graph]

            statistics_dict["input_graph"]["n_edges"].extend(input_graph_edges)
            statistics_dict["energy_graph"]["n_edges"].extend(energy_graph_edges)
            statistics_dict["input_graph"]["n_nodes"].extend(input_graph_nodes)
            statistics_dict["energy_graph"]["n_nodes"].extend(energy_graph_nodes)

        current_dataloader.smallest_n_edges_input_graph, current_dataloader.largest_n_edges_input_graph = get_x_smallest_and_largest(statistics_dict["input_graph"]["n_edges"], self.batch_size)
        current_dataloader.smallest_n_edges_energy_graph, current_dataloader.largest_n_edges_energy_graph = get_x_smallest_and_largest(statistics_dict["energy_graph"]["n_edges"], self.batch_size)
        current_dataloader.smallest_n_nodes_input_graph, current_dataloader.largest_n_nodes_input_graph = get_x_smallest_and_largest(statistics_dict["input_graph"]["n_nodes"], self.batch_size)
        current_dataloader.smallest_n_nodes_energy_graph, current_dataloader.largest_n_nodes_energy_graph = get_x_smallest_and_largest(statistics_dict["energy_graph"]["n_nodes"], self.batch_size)
        print("dataset statistics",mode, current_dataloader.smallest_n_edges_input_graph, current_dataloader.largest_n_edges_input_graph)




    def reinint_train_dataloader(self, epoch):

        def seed_worker(worker_id):
            worker_seed = (torch.initial_seed() + epoch) % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        generator = torch.Generator()
        generator.manual_seed(self.seed+ epoch)

        loader_kwargs = dict(
            batch_size=self.batch_size,
            collate_fn=self.pmap_collate,
            num_workers=self.num_workers,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=generator
        )
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
            loader_kwargs["persistent_workers"] = self.persistent_workers

        dataloader_train = DataLoader(self.dataset_train, **loader_kwargs)
        return dataloader_train


class SolutionDataset_InMemory(Dataset):
    def __init__(self, config = {}, dataset="ENZYMES", problem="MIS", mode="val", relaxed=False, seed=123):  ### TODO add orderign to config
        self.config = config
        self.dataset_name = dataset
        self.problem_name = problem
        self.mode = mode
        self.seed = seed
        self.relaxed = relaxed

        self.n_diffusion_steps = self.config["n_diffusion_steps"]+ 1
        self.buffer_size = 1000
        self.N_basis_states = self.config["N_basis_states"]

        self.get_dataset_paths(config, mode=mode, seed=seed)

        # Initialize cache for loaded data
        self._data_cache = {}
        # Optional: limit cache size to prevent memory issues
        self._max_cache_size = getattr(config, 'max_cache_size', 1000)
        #super().__init__(self.base_path, None, None, None)
        for i in range(len(self)): # init cache
            self.__getitem__(i)

    def get_dataset_paths(self, cfg, mode="", seed=None):
        if(self.problem_name == "MaxClv2"):
            select_data_name =  "MaxCl"
        else:
            select_data_name =  self.problem_name

        base_path = os.path.dirname(os.getcwd()) + "/DIffUCO/DatasetCreator/loadGraphDatasets/DatasetSolutions/"

        load_path = base_path + f"no_norm/{self.dataset_name}/{self.mode}/{self.seed}/{select_data_name}/indexed/"
        with open(load_path+ f"idx_{0}_solutions.pickle", "rb") as file:
            pickle.load(file)
        self.base_path = load_path

        self.val_mean_energy = 0.
        self.val_std_energy = 1.

        _, _, files = next(os.walk(load_path))
        file_count = len(files)
        self.n_graphs = 1 # file_count

    def __len__(self):
        return self.n_graphs

    def __getitem__(self, idx):
        idx += self.config["use_sample"]
        # Check if data is already cached
        if idx in self._data_cache:
            return self._data_cache[idx]

        # Load data from disk
        with open(self.base_path + f"idx_{idx}_solutions.pickle", "rb") as file:
            print("loading graph from HDD")
            graph_dict = pickle.load(file)

        input_graph = graph_dict["H_graphs"]

        if("U_net_graph_dict" in graph_dict.keys()):
            U_net_graph_dict = graph_dict["U_net_graph_dict"]
        else:
            U_net_graph_dict = None

        if("compl_H_graphs" in graph_dict.keys()):
            if( graph_dict["compl_H_graphs"] != None):
                energy_graphs = graph_dict["compl_H_graphs"]
            elif(type(graph_dict["compl_H_graphs"]) == list):
                if(len(graph_dict["compl_H_graphs"]) > 0):
                    energy_graphs = graph_dict["compl_H_graphs"]
            else:
                energy_graphs = input_graph
        else:
            if(self.problem_name == "MaxCl" or self.problem_name == "TSP" or self.problem_name == "MIS" or self.problem_name == "MaxClv2"):
                print(graph_dict.keys())
                raise ValueError("that is not possible")
            energy_graphs = input_graph

        # print("compare edges of input graph and energy graph", energy_graphs.edges.shape, input_graph.edges.shape)
        # print("compare edges of input graph and energy graph", energy_graphs.edges, input_graph.edges.shape)
        input_graph = GraphWithMeta(graph=input_graph.graph._replace(edges = input_graph.graph.edges.astype(np.float32)), meta=input_graph.meta)
        energy_graphs = input_graph#energy_graphs._replace(edges = energy_graphs.edges.astype(np.float32))

        return_dict = {"input_graph": input_graph, "energy_graph": energy_graphs, "energies": graph_dict["Energies"],
                       "U_net_graph_dict": U_net_graph_dict, "bs_bins": graph_dict["gs_bins"]}
        
        # Cache the result
        self._data_cache[idx] = return_dict
        
        # Optional: limit cache size (simple LRU-like behavior)
        if len(self._data_cache) > self._max_cache_size:
            # Remove the oldest entry (first key)
            oldest_key = next(iter(self._data_cache))
            del self._data_cache[oldest_key]
            
        return return_dict


def get_x_smallest_and_largest(lst, x):

    sorted_lst = sorted(lst)

    # Get the X smallest values
    x_smallest = sorted_lst[:x]

    # Get the X largest values
    x_largest = sorted_lst[-x:]

    return sum(x_smallest), sum(x_largest)
