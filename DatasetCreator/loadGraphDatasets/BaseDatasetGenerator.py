import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR)

from abc import ABC, abstractmethod
import jraph
import pickle
import numpy as np
from pathlib import Path
from DatasetCreator.jraph_utils import from_igraph_to_jgraph
from .save_utils import save_indexed_dict, load_indexed_dict
import igraph as ig
import networkx as nx

class BaseDatasetGenerator(ABC):
	"""
	Base Class for generating datasets
	"""
	def __init__(self, config):
		"""
		:param config: config file to specify the dataset
			{
				dataset_name: (string) name of the dataset,
				problem: (string) problem to be solved (e.g.: MaxCut, MDS, ...),
				mode: (string) [train, val, test],
				seed: (int) seed for random number generator,
				parent: (bool) whether to use parent directory,
				diff_ps: (bool) whether to use different p values,
				time_limit: (float) time limit for solving the problem (if applicable),
				n_graphs: (int) number of graphs to generate,
			}
		"""
		self.config = config

		self.seed = config["seed"]
		self.mode = config["mode"]
		self.save = config["save"]
		self.dataset_name = config["dataset_name"]
		self.problem = config["problem"]
		self.diff_ps = config["diff_ps"]
		# self.IsingFormulation = config["IsingFormulation"]
		self.time_limit = config.get("time_limit")

		# set path
		p = Path(os.getcwd())
		if config["parent"]:
			self.path = str(p.parent)
		else:
			self.path = str(p)

		# set seed
		if self.mode == "val":
			seed_int = 5
		elif self.mode == "test":
			seed_int = 4
		else:
			seed_int = 0
		np.random.seed(self.seed + seed_int)

	@abstractmethod
	def _generate_dataset(self) -> dict[str, list]:
		"""
		Generate the graph instances for the dataset

		- optionally override self.solve_graph(H_graph) if a solver is available
		- use self.save_instance_solution(indexed_solution_dict, idx) to save the graph instance
		- use self.save_solutions(solutions) to save the solutions
		"""
		raise NotImplementedError("generate_graph method not implemented")
	
	
	def generate_dataset(self) -> dict[str, list]:
		solutions = self._generate_dataset()

		for idx in range(len(solutions["Energies"])):
			indexed_solution_dict = {}
			for key in solutions.keys():
				if len(solutions[key]) > 0:
					indexed_solution_dict[key] = solutions[key][idx]
			self.save_instance_solution(indexed_solution_dict, idx)
		self.save_solutions(solutions)
		return solutions
	

	def solve_graph(self, H_graph, g) -> (float, float, list, float, jraph.GraphsTuple):
		"""
		Optional solver hook. Subclasses can override to provide exact/heuristic solutions.

		:param H_graph: jraph graph instance
		:param g: igraph graph instance
		:return: (Energy, boundEnergy, solution, runtime, H_graph_compl)
		"""
		
		# Default to placeholder values when no solver is provided.
		Energy = 0.
		boundEnergy = 0.
		solution = np.ones_like(H_graph.nodes)
		runtime = None

		H_graph_compl = None
		return Energy, boundEnergy, solution, runtime, H_graph_compl

	def igraph_to_jraph(self, g: ig.Graph) -> (jraph.GraphsTuple, float, int):
		"""
		Convert igraph graph to jraph graph

		:param g: igraph graph
		:return: (H_graph, density, graph_size)
		"""
		density = 2 * g.ecount() / (g.vcount() * (g.vcount() - 1))
		graph_size = g.vcount()
		return from_igraph_to_jgraph(g), density, graph_size

	def nx_to_jraph(self, gnx: nx.Graph) -> (jraph.GraphsTuple, float, int):
		"""
		Convert networkx graph to jraph graph via igraph

		:param gnx: networkx graph
		:return: (H_graph, density, graph_size)
		"""
		g = ig.Graph.TupleList(gnx.edges(), directed=False)
		density = 2 * g.ecount() / (g.vcount() * (g.vcount() - 1))
		graph_size = g.vcount()
		return from_igraph_to_jgraph(g), density, graph_size

	def nx_to_igraph(self, gnx: nx.Graph) -> ig.Graph:
		"""
		Convert networkx graph to igraph graph

		:param gnx: networkx graph
		:return: igraph graph
		"""
		return ig.Graph.TupleList(gnx.edges(), directed=False)

	def save_instance_solution(self, indexed_solution_dict, idx):
		"""
		Save the graph instance solution to a file

		:param indexed_solution_dict: dictionary containing the solution
		:param idx: index of the solution
		"""
		indexed_solution_dict["time_limit"] = self.time_limit
		save_indexed_dict(path=self.path, mode=self.mode, dataset_name=self.dataset_name, i=idx,
						EnergyFunction=self.problem, seed=self.seed, indexed_solution_dict=indexed_solution_dict)

	def save_solutions(self, solutions):
		"""
		save the solutions to a file

		:param solutions: dictionary containing the solutions
		"""
		if self.save:
			new_path = self.path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{self.dataset_name}"
			if not os.path.exists(new_path):
				os.makedirs(new_path)

			save_path = self.path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{self.dataset_name}/{self.mode}_{self.problem}_seed_{self.seed}_solutions.pickle"
			pickle.dump(solutions, open(save_path, "wb"))


	def load_solutions(self):
		solutions_list = []
		for idx in range(self.graph_config[f"n_{self.mode}"]):
			solutions_list.append( load_indexed_dict(path=self.path, mode=self.mode, dataset_name=self.dataset_name, i=idx,
							EnergyFunction=self.problem, seed=self.seed))

		avrg_runtimes = np.sum([el["runtimes"] for el in solutions_list])
		avrg_Energies = np.mean([el["Energies"] for el in solutions_list])

		return_dict = {"runtimes": avrg_runtimes,"Energies": avrg_Energies}
		print("number of graphs", len(solutions_list))
		return return_dict
