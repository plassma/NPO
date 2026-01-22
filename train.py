import os.path
import copy
import numpy as np
from functools import partial
import jax
from jax import lax
import jax.numpy as jnp
import pickle
import optax
import jraph
from tqdm import tqdm
import wandb
from matplotlib import pyplot as plt
from DatasetCreator.loadGraphDatasets.HCPDatasetGenerator import plot
from NoiseDistributions import get_Noise_class
from Trainers import get_Trainer_class
from Networks.DiffModel import DiffModel
from utils.lr_schedule import cos_schedule
from EnergyFunctions import get_Energy_class
from Data.LoadGraphDataset import SolutionDatasetLoader
from jax.tree_util import tree_flatten
import time
import jraph_utils
from utils import reshape_utils
import os
import tempfile
import warnings
import uuid


class TrainMeanField:
	def __init__(self, config, load_wandb_id = None, eval_step_factor = 1, load_best_parameters = False):
		self.load_wandb_id = load_wandb_id
		self.load_step = config["load_step"]
		self.load_best_parameters = load_best_parameters
		jax.config.update('jax_disable_jit', not config["jit"])

		self.path_to_models = os.getcwd() + "/Checkpoints"

		self.config = self._init_config(config)
		self.config.setdefault("ownership_weight", 1.0)
		self.ownership_weight = self.config["ownership_weight"]
		self.config["eval_step_factor"] = eval_step_factor
		print(self.config)

		self.seed = self.config["seed"]
		self.key = jax.random.PRNGKey(self.seed)

		# if epoch % save_modulo == 0 the params will be saved
		self.save_modulo = 100
		self.save_checkpoints = bool(self.config.get("save_checkpoints", True))

		self.dataset_name = self.config["dataset_name"]
		self.problem_name = self.config["problem_name"]

		if(self.problem_name == "TSP"):
			self.pad_delta = 0.
		elif("large" in self.dataset_name):
			self.pad_delta = 0.05
		else:
			self.pad_delta = 0.3
		self.pad_k = 1. + self.pad_delta*(30./self.config["batch_size"])*len(jax.devices())

		self.grid_num = min([int(12*self.config["batch_size"]/(30*len(jax.devices()))),7])
		self.edge_grid_factor = 2
		# if(len(jax.devices()) > 1):
		# 	self.pad_k = 2.

		print("pad_k is", self.pad_k, "grid num", self.grid_num)

		self.epochs = self.config["N_warmup"] + self.config["N_anneal"] + self.config["N_equil"]
		self.config["epochs"] = self.epochs

		self.lr = self.config["lr"]
		self.min_lr = self.config.get("min_lr", self.lr/3)
		self.N_basis_states = self.config["N_basis_states"]

		if("AnnealSchedule" not in self.config.keys()):
			self.config["AnnealSchedule"] = "linear"
			self.AnnealSchedule = self.config["AnnealSchedule"]
		else:
			self.AnnealSchedule = self.config["AnnealSchedule"]

		if("lr_schedule" not in self.config.keys()):
			self.config["lr_schedule"] = "cosine"
			self.lr_schedule = self.config["lr_schedule"]
		else:
			self.lr_schedule = self.config["lr_schedule"]

		self.batch_size = self.config["batch_size"]
		self.energy_per_node_dim = 1

		self.relaxed = self.config["relaxed"]

		self.T_max = self.config["T_max"]

		
		self.T = self.T_max
		self.N_warmup = self.config["N_warmup"]
		self.N_anneal = self.config["N_anneal"]
		self.N_equil = self.config["N_equil"]
		self.loss_alpha = self.config["loss_alpha"]
		self.MCMC_steps = self.config["MCMC_steps"]

		self.n_diffusion_steps = self.config["n_diffusion_steps"]
		self.mode = self.config["mode"]
		self.beta_factor = self.config["beta_factor"]

		# Network

		if(self.problem_name == "TSP"):
			self.config["edge_updates"] = True
			if("20" in self.dataset_name):
				self.n_bernoulli_features = 20
			elif("100" in self.dataset_name):
				self.n_bernoulli_features = 100
		elif self.problem_name == "HCP":
			self.n_bernoulli_features = config["n_bernoulli_features"] # todo plassma: hardcoded for now
		else:
			self.n_bernoulli_features = 2

		self.config["n_bernoulli_features"] = self.n_bernoulli_features

		if(self.problem_name == "TSP"):
			self.config["n_features_list_prob"] = [120,120,self.n_bernoulli_features]
		elif(self.problem_name == "IsingModel"):
			self.config["n_features_list_prob"] = [64,64,self.n_bernoulli_features]
		else:		
			self.config["n_features_list_prob"] = [120,64,2]

		self.n_features_list_prob = self.config["n_features_list_prob"]
		self.config["n_bernoulli_features"] = self.n_bernoulli_features

		self.n_features_list_nodes = self.config["n_features_list_nodes"]
		self.n_features_list_edges = self.config["n_features_list_edges"]
		self.n_features_list_messages = self.config["n_features_list_messages"]
		self.n_features_list_encode = self.config["n_features_list_encode"]
		self.n_features_list_decode = self.config["n_features_list_decode"]
		self.message_passing_weight_tied = self.config["message_passing_weight_tied"]
		self.linear_message_passing = self.config["linear_message_passing"]

		self.transformer_type = self.config.get("transformer_type", "linear")
		self.config["transformer_type"] = self.transformer_type

		if("bfloat16" in self.config.keys()):
			self.bfloat16 = self.config["bfloat16"]
		else:
			self.config["bfloat16"] = False
			self.bfloat16 = self.config["bfloat16"]

		if("sampling_temp" in self.config.keys()):
			pass
		else:
			self.config["sampling_temp"] = 1.

		if("n_sampling_rounds" in self.config.keys()):
			pass
		else:
			self.config["n_sampling_rounds"] = 1.

		if("T_target" in self.config.keys()):
			self.T_target = self.config["T_target"]
		else:
			self.config["T_target"] = 0.
			self.T_target = self.config["T_target"]

		if("n_test_basis_states" in self.config.keys()):
			self.N_test_basis_states = self.config["n_test_basis_states"]
		else:
			self.config["n_test_basis_states"] = 8
			self.N_test_basis_states = self.config["n_test_basis_states"]

		if("edge_updates" in self.config.keys()):
			self.edge_updates = self.config["edge_updates"]
		else:
			self.edge_updates = True

		if("time_encoding" in self.config.keys()):
			self.time_encoding = self.config["time_encoding"]
		else:
			self.config["time_encoding"] = "one_hot"
			self.time_encoding = self.config["time_encoding"]

		if("mean_aggr" in self.config.keys()):
			self.mean_aggr = self.config["mean_aggr"]
		else:
			self.mean_aggr = False

		if("noise_potential" in self.config.keys()):
			self.noise_potential = self.config["noise_potential"]
		else:
			self.noise_potential = "annealed_obj"

		if("project_name" in self.config.keys()):
			self.project_name = self.config["project_name"]
		else:
			self.project_name = ""

		if("grad_clip" in self.config.keys()):
			self.grad_clip = self.config["grad_clip"]
		else:
			self.grad_clip = True
			self.config["grad_clip"] = self.grad_clip

		if ("TD_k" in self.config.keys()):
			pass
		else:
			self.config["TD_k"] = 3

		if ("value_weighting" in self.config.keys()):
			pass
		else:
			self.config["value_weighting"] = 0.65

		if ("clip_value" in self.config.keys()):
			pass
		else:
			self.config["clip_value"] = 0.2

		if ("time_conditioning" in self.config.keys()):
			self.time_conditioning = self.config["time_conditioning"]
		else:
			self.time_conditioning = False

		self.use_wandb = bool(self.config.get("wandb", False))
		self.wandb_mode = "online" if self.use_wandb else "disabled"

		# self.wandb_mode = "disabled"

		config = self.config

		self.wandb_project = f"{self.project_name}{config['mode']}_{config['dataset_name']}_{config['problem_name']}_relaxed_{config['relaxed']}_deeper" + ("_loaded" if self.load_wandb_id != None else "")
		if config['T_max'] > 0.:
			self.wandb_group = f"{config['seed']}_LMP_T_{config['T_max']}_noise_potential_{config['noise_potential']}_anneal_{config['N_anneal']}"
		else:
			self.wandb_group = f"{config['seed']}_LMP_T_{config['T_max']}_anneal_{config['N_anneal']}"

		wandb_run = f"sample_{config['use_sample']}_emb_{config['embedding_dim']}_n_diff_{config['n_diffusion_steps']}_T_{config['T_max']:.6f}"

		if self.use_wandb:
			self.wandb_run_id = wandb.util.generate_id()
		else:
			self.wandb_run_id = uuid.uuid4().hex
		self.wandb_run = f"{self.load_wandb_id}_{self.wandb_run_id}_{wandb_run}"

		self.best_rel_error = float('inf')
		self.best_energy = float("inf")

		self.stop_epochs = self.config["stop_epochs"]
		self.epochs_since_best = 0

		self.__init_Energy_functions()
		self.__init_noise_distribution_class()
		self.config.pop("vmapped_energy_loss_func")
		self.config.pop("vmapped_energy_func")
		self.__init_dataset()
		self.__init_network()
		self.__init__Trainer()
		self.__init_optimizer_and_params()
		self.__init_functions()
		self.__init_wandb(self.config)
		self.last_eval_log = None
		#self.__init_beta_list()

	def __init__Trainer(self):
		TrainerClass_func = get_Trainer_class(self.config)
		self.TrainerClass = TrainerClass_func(self.config, self.EnergyClass, self.NoiseDistrClass, self.model)

	def __init_Energy_functions(self):
		EnergyClass = get_Energy_class(self.config)
		self.EnergyClass = EnergyClass
		self.relaxed_energy = EnergyClass.calculate_Energy
		self.relaxed_Energy_for_Loss = EnergyClass.calculate_Energy_loss

		def _relaxed_energy(meta_graph, bins, node_gr_idx, ownership_weight):
			if self.problem_name == "HCP":
				return self.relaxed_energy(meta_graph, bins, node_gr_idx, A=ownership_weight)
			return self.relaxed_energy(meta_graph, bins, node_gr_idx)

		def _relaxed_energy_loss(meta_graph, logits, node_gr_idx, ownership_weight):
			if self.problem_name == "HCP":
				return self.relaxed_Energy_for_Loss(meta_graph, logits, node_gr_idx, A=ownership_weight)
			return self.relaxed_Energy_for_Loss(meta_graph, logits, node_gr_idx)

		_vmapped_relaxed_energy = jax.vmap(_relaxed_energy, in_axes=(None, 1, None, None), out_axes=(1))
		_vmapped_relaxed_energy_for_Loss = jax.vmap(_relaxed_energy_loss, in_axes=(None, 1, None, None),
														out_axes=(1))

		def vmapped_relaxed_energy(meta_graph, bins, node_gr_idx, ownership_weight=None):
			weight = self.ownership_weight if ownership_weight is None else ownership_weight
			return _vmapped_relaxed_energy(meta_graph, bins, node_gr_idx, weight)

		def vmapped_relaxed_energy_for_Loss(meta_graph, logits, node_gr_idx, ownership_weight=None):
			weight = self.ownership_weight if ownership_weight is None else ownership_weight
			return _vmapped_relaxed_energy_for_Loss(meta_graph, logits, node_gr_idx, weight)

		self.vmapped_relaxed_energy = vmapped_relaxed_energy
		self.vmapped_relaxed_energy_for_Loss = vmapped_relaxed_energy_for_Loss
		self.config["vmapped_energy_loss_func"] = self.vmapped_relaxed_energy_for_Loss
		self.config["vmapped_energy_func"] = self.vmapped_relaxed_energy

	def __init_noise_distribution_class(self):
		self.NoiseDistrClass = get_Noise_class(self.config)


	def _load_checkpoint_file(self, file_name):
		path_folder = f"{self.path_to_models}/{self.load_wandb_id}/"
		with open(os.path.join(path_folder, file_name), "rb") as f:
			return pickle.load(f)

	def _load_checkpoint(self, kind):
		wandb_run_id = self.load_wandb_id
		if kind == "last":
			file_name = f"{wandb_run_id}_last_epoch.pickle"
		elif kind == "best":
			file_name = f"best_{wandb_run_id}.pickle"
		elif kind == "step":
			file_name = f"{wandb_run_id}_{self.load_step}.pickle"
		else:
			raise ValueError(f"Unknown checkpoint kind: {kind}")

		loaded = self._load_checkpoint_file(file_name)
		if isinstance(loaded, dict):
			return loaded

		params = loaded[0]
		config = loaded[1] if not self.config["load_only_params"] else self.config
		epoch = None
		if kind != "last":
			try:
				epoch = self._load_checkpoint("last")["epoch"]
			except Exception:
				epoch = 0
		return {"params": params, "config": config, "epoch": epoch}

	def _init_config(self, config):
		if(self.load_wandb_id == None or config["load_only_params"]):
			return config
		else:
			loaded_dict = self._load_checkpoint("last")
			loaded_config = loaded_dict["config"]
			loaded_config["N_anneal"] = max(config["N_anneal"], loaded_config.get("N_anneal", 0))
			return loaded_config

	def __init_network(self):
		"""
		initialize network and optimizer
		"""
		self.graph_mode = self.config["graph_mode"]

		if("graph_norm" in self.config.keys()):
			self.graph_norm = self.config["graph_norm"]
		else:
			self.graph_norm = False

		self.model = DiffModel(n_features_list_prob=self.n_features_list_prob,
								n_features_list_nodes=self.n_features_list_nodes,
								n_features_list_edges=self.n_features_list_edges,
								n_features_list_messages=self.n_features_list_messages,
								n_features_list_encode=self.n_features_list_encode,
								n_features_list_decode=self.n_features_list_decode,
								n_diffusion_steps = self.n_diffusion_steps,
							   time_encoding = self.time_encoding,
								n_diff_steps = self.n_diffusion_steps,
							   message_passing_weight_tied=self.message_passing_weight_tied,
								linear_message_passing=self.linear_message_passing,
								edge_updates = self.edge_updates,
								problem_type = self.problem_name,
								n_bernoulli_features = self.n_bernoulli_features,
								mean_aggr = self.mean_aggr,
							   EncoderModel = self.graph_mode, dim_energy_per_node = self.energy_per_node_dim,
							   train_mode = self.config["train_mode"],
							   graph_norm = self.graph_norm, bfloat16 = self.bfloat16, dataset_name = self.dataset_name, embedding_dim=self.config["embedding_dim"], node_emb_type=self.config["node_emb_type"],
							   augment_rooms=self.config["augment_rooms"],
							   node_transformer_num_layers=self.config.get("node_transformer_num_layers", 4),
							   node_transformer_num_heads=self.config.get("node_transformer_num_heads", 4),
							   node_transformer_dropout_rate=self.config.get("node_transformer_dropout_rate", 0.05),
							   transformer_type=self.transformer_type)


	def __init_optimizer_and_params(self):
		if(self.load_wandb_id == None):
			self.curr_epoch = 0
			self.__init_params()
			self.__init_optimizer(self.lr, self.params)
		else:
			if(self.load_best_parameters):
				print("Best Parameters are Loaded!")
				loaded_dict = self._load_checkpoint("best")
			elif self.config["load_step"] > 0:
				loaded_dict = self._load_checkpoint("step")
			else:
				if(self.config["train_mode"] in ["PPO", "GRPO"] and self.config["problem_name"] != "IsingModel"):
					try:
						loaded_dict = self._load_checkpoint("best")
					except Exception:
						loaded_dict = self._load_checkpoint("last")
				else:
					loaded_dict = self._load_checkpoint("last")

			self.loaded_checkpoint = loaded_dict
			print("loaded dict", self.load_best_parameters, loaded_dict.keys())
			self.curr_epoch = loaded_dict["epoch"] if not self.config["load_only_params"] else 0
			self.params = loaded_dict["params"]

			self.__init_optimizer(self.lr, self.params)
			self.__init_params()

		if(self.bfloat16):
			print("cast to bfloat16 init")
			#print(jax.tree_util.tree_map(lambda x: x.dtype, self.params))
			self.params = jax.tree_util.tree_map(lambda x: x.astype(jax.numpy.bfloat16), self.params)
			#print(jax.tree_util.tree_map(lambda x: x.dtype, self.params))

	def __init_optimizer(self, lr, params):
		# self.optimizer = optax.radam(learning_rate=self.curr_lr)
		self.epoch_length = len(self.dataloader_train)*self.TrainerClass.inner_update_steps
		if(self.lr_schedule == "cosine"):
			lr_func = cos_schedule
		else:

			def get_lr(step, epochs, min_lr = None, max_lr = None):
				return lr

			lr_func = get_lr

		self.lr_func = lr_func
		if(self.config["grad_clip"]):
			opt = optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_radam(),
										 optax.scale_by_schedule(lambda step: -lr_func(step, self.epoch_length*(self.N_anneal + self.N_warmup + self.N_equil), max_lr=lr, min_lr = self.min_lr)))
			opt_init, self.opt_update = opt

		else:
			# opt_init, self.opt_update = optax.chain(optax.clip(1.0), optax.scale_by_radam(),
			# 							 optax.scale(-lr))

			# optimizer = optax.adam(learning_rate=lr)
			# self.opt_update = optimizer.update
			# opt_init = optimizer.init

			opt_init, self.opt_update = optax.chain( optax.scale_by_radam(),
										 optax.scale_by_schedule(lambda step: -lr_func(step, self.epoch_length*(self.N_anneal + self.N_warmup + self.N_equil), max_lr=lr, min_lr = self.min_lr)))

		if(self.load_wandb_id == None or self.config["load_only_params"]):
			self.opt_state = jax.pmap(opt_init)(params)

		else:
			loaded_dict = getattr(self, "loaded_checkpoint", None) or self._load_checkpoint("last")
			opt_state = loaded_dict.get("opt_state")
			if opt_state is None:
				opt_state = self._load_checkpoint("last")["opt_state"]
			self.opt_state = opt_state
			self.opt_state = jax.tree_util.tree_map(lambda x: x[0], self.opt_state)
			self.opt_state = jax.device_put_replicated(self.opt_state, list(jax.devices()))

		self.TrainerClass.opt_update = self.opt_update

	def __update_lr(self, epoch):
		lr = self.lr_func(epoch, self.N_anneal+ self.N_warmup, max_lr=self.lr, min_lr = self.lr/10)
		return lr

	def __init_functions(self):
		"""
		initialize functions (for jitting or vmapping)
		"""
		pass

	def __init_params(self):
		"""
		initialize network parameters
		"""
		self.key, subkey = jax.random.split(self.key)
		jraph_graph_dict = next(iter(self.dataloader_val))

		if (self.load_wandb_id != None):
			self.params = jax.tree_util.tree_map(lambda x: x[0], self.params)
		elif(self.graph_mode != "U_net"):

			input_graph_list, energy_graphs = self._prepare_graphs(jraph_graph_dict, mode = "val")

			batched_graph = input_graph_list["graphs"][0]
			X_prev = jnp.ones((batched_graph.nodes.shape[1], 1))
			energy_per_node_dim = jnp.ones((batched_graph.nodes.shape[1], self.energy_per_node_dim))

			input_graph_list = {"graphs": [jax.tree_util.tree_map(lambda x: x[0], input_graph_list["graphs"][0])]}
			t_idx_per_node = jnp.ones((batched_graph.nodes.shape[1],1))
			self.params = self.model.init({"params": subkey}, input_graph_list, X_prev, energy_per_node_dim, t_idx_per_node, subkey)
		elif(self.graph_mode == "U_net"):
			reps = 10
			iters = len(self.dataloader_train)*reps
			U_net_graph_dict = jraph_graph_dict["U_net_graph_dict"][0]
			U_net_graph_dict = jax.tree_util.tree_map(lambda x: jnp.array(x), U_net_graph_dict)
			print(jax.tree_util.tree_map(lambda x: x.shape, U_net_graph_dict))

			input_graph_list, energy_graphs = self._prepare_graphs(jraph_graph_dict)

			X_prev = jnp.ones((U_net_graph_dict["graphs"][0].nodes.shape[0], 1))
			rand_node_features = jnp.ones((U_net_graph_dict["graphs"][0].nodes.shape[0], self.energy_per_node_dim))
			self.params = self.model.init({"params": subkey}, U_net_graph_dict, X_prev,rand_node_features, 0, subkey)

		else:
			raise ValueError("")


		num_gpus = jax.local_device_count()
		print("Training is distributed across ", num_gpus, "devices!")
		# if(num_gpus <= 1):
		# 	pass
		# else:
		self.params = jax.device_put_replicated(self.params, list(jax.devices()))

		print("pmapped params")
		print(jax.tree_util.tree_map(lambda x: x.shape, self.params))

	def __init_dataset(self):
		self.data_generator = SolutionDatasetLoader(config = self.config, dataset=self.dataset_name,
											   problem=self.problem_name,
											   batch_size=self.batch_size,
											   relaxed=self.relaxed,
											   seed=self.seed)
		self.dataloader_train, self.dataloader_test, self.dataloader_val, (
			self.mean_energy, self.std_energy) = self.data_generator.dataloaders()

	def __init_wandb(self, config):
		"""
		initialize weights and biases

		@param project: project name
		"""
		if not self.use_wandb:
			return
		if(self.config["wandb"]):
			wandb.init(project=self.wandb_project, name=self.wandb_run, group=self.wandb_group, id=self.wandb_run_id,
				   config=config, mode=self.wandb_mode, settings=wandb.Settings(_service_wait=300))



	def __linear_annealing(self, epoch):
		if epoch < self.N_warmup:
			T_curr = self.T_max
		elif epoch >= self.N_warmup and epoch < self.epochs - self.N_equil - 1:
			progress = (epoch - self.N_warmup) / max(self.N_anneal, 1)
			T_curr = self.T_max + (self.T_target - self.T_max) * progress
		else:
			T_curr = self.T_target

		return T_curr

	def __exp_annealing(self, epoch):
		if (epoch < self.N_warmup):
			T_curr = self.T_max
		elif(epoch >= self.N_warmup and epoch <= self.epochs - self.N_equil - 1):
			factor = 4000
			T_curr = self.T_target*1/(1- 0.998**(factor*((epoch - self.N_warmup +1 )/self.epochs )))
		else:
			T_curr = self.T_target

		return T_curr
	
	def __powerlaw_annealing(self, epoch, power):
		if epoch < self.N_warmup:
			T_curr = self.T_max
		elif epoch >= self.N_warmup and epoch < self.epochs - self.N_equil - 1:
			T_curr = max([self.T_target + (self.T_max - self.T_target) * ((self.N_anneal - epoch) / self.N_anneal)**power, 0])
		else:
			T_curr = self.T_target

		return T_curr

	def __piecewise_linear_annealing(self, epoch):
		schedule = self.config.get("piecewise_linear_anneal_schedule", [])
		if not schedule:
			raise ValueError("piecewise_linear_anneal_schedule is empty or missing")
		schedule = sorted(schedule, key=lambda item: item[0])
		if epoch <= schedule[0][0]:
			return schedule[0][1]
		if epoch >= schedule[-1][0]:
			return schedule[-1][1]
		for (start_step, start_temp), (end_step, end_temp) in zip(schedule[:-1], schedule[1:]):
			if start_step <= epoch <= end_step:
				if end_step == start_step:
					return end_temp
				progress = (epoch - start_step) / (end_step - start_step)
				return start_temp + (end_temp - start_temp) * progress
		return schedule[-1][1]


	def _temperature_from_schedule(self, epoch):
		if ("linear" == self.AnnealSchedule):
			return self.__linear_annealing(epoch), False
		elif ("exp" == self.AnnealSchedule):
			return self.__exp_annealing(epoch), False
		elif ("powerlaw" == self.AnnealSchedule):
			return self.__powerlaw_annealing(epoch, self.config["powerlaw_exponent"]), False
		elif ("piecewise_linear" == self.AnnealSchedule):
			return self.__piecewise_linear_annealing(epoch), False
		elif ("linear_cyclic" == self.AnnealSchedule):
			cycle_length = self.config["anneal_cycle_length"]
			if epoch > self.N_warmup + self.N_anneal:
				return self.T_target, True
			cycle_epoch = epoch % cycle_length
			return self.__linear_annealing(cycle_epoch), False
		else:
			raise ValueError("schedule not implemented")


	def train_step(self, batch_dict):
		### TODO add code that switches of the buffer
		step1 = time.time()
		graph_batch, energy_graph_batch = self._prepare_graphs(batch_dict, mode = "train")
		step2 = time.time()
		batching_time = step2 - step1

		self.params, self.opt_state, loss, (log_dict, energy_graph_batch, self.key) = self.TrainerClass.train_step(self.params, self.opt_state, graph_batch,
																							  energy_graph_batch, self.T, self.key)

		return loss, (log_dict, energy_graph_batch, batching_time)

	def train(self, max_epochs=None, max_batches=None, eval_every=100, return_metrics=False):

		if self.use_wandb:
			wandb.define_metric("train/metrics")
			wandb.define_metric("train/loss" )
			wandb.define_metric("train/loss" )

		print("first evaluation...")
		self.save_metrics_dict = {}
		self.save_metrics_dict["eval/energy"] = []
		self.save_metrics_dict["eval/gt_energy"] = []
		self.save_metrics_dict["eval/rel_error"] = []
		self.__save_checkpoint(f"{self.wandb_run_id}_last_epoch.pickle", epoch=0)
		self.last_eval_log = self.eval(epoch=0)
		print("start training...")
		epoch_start = self.curr_epoch
		epoch_end = self.epochs
		if max_epochs is not None:
			epoch_end = min(epoch_end, epoch_start + max_epochs)
		epoch_range = np.arange(epoch_start, epoch_end)
		print("start training for ", self.epochs, self.curr_epoch)
		#graph_shape_list = []
		last_epoch = None
		for epoch in tqdm(epoch_range, desc="Training"):
			last_epoch = epoch
			print("epoch", epoch, "in", self.epochs)
			start_train_time = time.time()

			self.T, skip_epoch = self._temperature_from_schedule(epoch)
			self.ownership_weight = 1#0.1 + epoch * 2 / self.epochs
			if skip_epoch:
				continue

			### TODO move code that updates MCMC buffer to this palce and update the MCMC buffer for a larger batchsize
			step4 = time.time()
			wandb_log_dict = {}
			epoch_time_dict = {}
			epoch_time_dict["epoch_time/dataloader"] = []
			epoch_time_dict["epoch_time/batching"] = []
			epoch_time_dict["epoch_time/backprob"] = []
			epoch_time_dict["epoch_time/logging"] = []
			for iter, (batch_dict) in enumerate(self.dataloader_train):
				if max_batches is not None and iter >= max_batches:
					break
				gt_normed_energies = batch_dict["energies"]
				print("batch", iter, "of", len(self.dataloader_train))
				print("batchsize is", len(gt_normed_energies))

				step1 = time.time()
				loss, (log_dict, energy_graph_batch, batching_time) = self.train_step(batch_dict)
				step3 = time.time()

				if("metrics" in log_dict.keys()):
					log_dict_metrics = jax.tree_util.tree_map(reshape_utils.unravel_dict, log_dict["metrics"])
					batch_log_dict = self.__calculate_reporting(energy_graph_batch.graph,
						log_dict_metrics["energies"], gt_normed_energies, log_dict_metrics["spin_log_probs"], log_dict_metrics["free_energies"])
					batch_log_dict["solution_prob_mean"] = log_dict_metrics["solution_prob_mean"]
					batch_log_dict["solution_prob_min"] = log_dict_metrics["solution_prob_min"]

					### concatenate along device dim
					energy_dict = {f"energies/{key}": log_dict["energies"][key] for key in log_dict["energies"]}
					for key in batch_log_dict.keys():
						if key not in wandb_log_dict:
							wandb_log_dict[key] = []

						wandb_log_dict[key].append(batch_log_dict[key][None, ...])

					for key in energy_dict.keys():
						if key not in wandb_log_dict:
							wandb_log_dict[key] = []
						if key.startswith("energies/energy"):
							wandb_log_dict[key + "_min"] = []
							wandb_log_dict[key + "_max"] = []
							wandb_log_dict[key + "_std"] = []
							wandb_log_dict[key + "_min"].append(energy_dict[key].min())
							wandb_log_dict[key + "_max"].append(energy_dict[key].max())
							wandb_log_dict[key + "_std"].append(energy_dict[key].std())

						wandb_log_dict[key].append(energy_dict[key])

				loss_dict = {f"losses/{key}": log_dict["Losses"][key] for key in log_dict["Losses"]}
				for key in loss_dict.keys():
					if key not in wandb_log_dict:
						wandb_log_dict[key] = []
					wandb_log_dict[key].append(loss_dict[key])

				if("time" in log_dict.keys()):
					loss_dict = {f"time/{key}": np.sum(log_dict["time"][key]) for key in log_dict["time"]}
					for key in loss_dict.keys():
						if key not in wandb_log_dict:
							wandb_log_dict[key] = []
						wandb_log_dict[key].append(loss_dict[key])

				epoch_time_dict["epoch_time/dataloader"].append(step1-step4)
				step4 = time.time()
				epoch_time_dict["epoch_time/backprob"].append(step3 - step1)
				epoch_time_dict["epoch_time/batching"].append(batching_time)
				epoch_time_dict["epoch_time/logging"].append(step4 - step3)

			print("backprobagation time", np.sum(epoch_time_dict["epoch_time/backprob"]))
			print("logging time", np.sum(epoch_time_dict["epoch_time/logging"]))
			print("dataloader time", np.sum(epoch_time_dict["epoch_time/dataloader"]))
			print("batching time", np.sum(epoch_time_dict["epoch_time/batching"]))

			end_train_time = time.time()
			train_time_needed = end_train_time - start_train_time

			new_lr = np.mean(self.lr_func(self.opt_state[1].count, self.epoch_length * (self.N_anneal + self.N_warmup+ + self.N_equil), max_lr=self.lr, min_lr=self.min_lr))

			train_log_dict = {
				"train/epoch": epoch,
				"schedules/lr": new_lr,
				"schedules/T": self.T,
				"schedules/time": train_time_needed,
				"schedules/ownership_weight": self.ownership_weight

			}

			wandb_epoch_time_dict = {}
			for key in epoch_time_dict.keys():
				wandb_epoch_time_dict["train/" + key] = np.sum(epoch_time_dict[key])

			for key in wandb_log_dict.keys():
				#print(type(wandb_log_dict[key]))
				try:
					#print(key)
					#print("shape before concatenation", np.array(wandb_log_dict[key]).shape)
					### concatenate along
					res = np.concatenate(np.concatenate(wandb_log_dict[key], axis=1), axis=0)
					#print(res.shape, "concate result")
					train_log_dict["train/" + key] = np.mean(res)
				except:

					#print("shape jsut calc the mean", np.array(wandb_log_dict[key]).shape)
					train_log_dict["train/" + key] = np.mean(wandb_log_dict[key])

			combined_train_log = {**train_log_dict, **wandb_epoch_time_dict}
			if self.use_wandb:
				wandb.log(combined_train_log, step=epoch)

			if eval_every is not None and (epoch + 1) % eval_every == 0:
				self.last_eval_log = self.eval(epoch=epoch + 1)

			if self.epochs_since_best == self.stop_epochs:
				# early stopping
				print("run stopped due to break condition")
				break
		if return_metrics and (self.last_eval_log is None or (eval_every is None) or (last_epoch is not None and (last_epoch + 1) % eval_every != 0)):
			eval_epoch = (last_epoch + 1) if last_epoch is not None else self.curr_epoch
			self.last_eval_log = self.eval(epoch=eval_epoch)

		if self.use_wandb:
			wandb.finish()
		if return_metrics:
			return self.last_eval_log

	def sample(self,N = 4000):
		dataloader = self.dataloader_val
		self.TrainerClass.N_test_basis_states = N
		best_so_far = np.inf
		for iter, (batch_dict) in enumerate(dataloader):
			graph_batch, energy_graph_batch = self._prepare_graphs(batch_dict, mode = "eval")

			self.key, subkey = jax.random.split(self.key)
			batched_key = jax.random.split(subkey, num = len(jax.devices()))

			loss, (log_dict, _) = self.TrainerClass.pmap_sample(self.params, graph_batch, energy_graph_batch, self.T, batched_key)

			vals = log_dict["X_0"][0, :, :, 0]
			log_dict["graph_batch"] = graph_batch
			unique_solutions = np.unique(vals, axis=1)
			energies = log_dict["metrics"]["energies"].squeeze()
			best_so_far = min(best_so_far, energies.min())
			print(f"Mean energy is {np.mean(energies)}, min energy is {np.min(energies)}, max energy is {np.max(energies)}, best so far is {best_so_far}")
			print(f"Number of unique solutions is {unique_solutions.shape[0]}")

		return log_dict


	def eval(self, epoch, mode = "eval"):

		dataloader = self.dataloader_val

		wandb_log_dict = {}
		save_metrics_at_epoch = {}
		save_metrics_at_epoch["eval/energy"] = []
		save_metrics_at_epoch["eval/gt_energy"] = []
		save_metrics_at_epoch["eval/rel_error"] = []
		for iter, (batch_dict) in enumerate(dataloader):
			gt_normed_energies = batch_dict["energies"]
			print("batchsize is", len(gt_normed_energies))

			graph_batch, energy_graph_batch = self._prepare_graphs(batch_dict, mode = mode)

			self.key, subkey = jax.random.split(self.key)
			batched_key = jax.random.split(subkey, num = len(jax.devices()))

			loss, (log_dict, _) = self.TrainerClass.evaluation_step(self.params, graph_batch, energy_graph_batch, self.T, batched_key, mode = mode, epoch = epoch, epochs = self.epochs)

			if self.use_wandb:
				with tempfile.NamedTemporaryFile(suffix=".png") as target:
					plot(None, graph_batch["graphs"][0].graph.globals["node_types"][0, :-1],target.name, solution_nodes=log_dict["X_0"][0, :-1, 0, 0], meta_graph=graph_batch["graphs"][0])
					wandb.log({"random sample": wandb.Image(target.name)}, commit=False, step=epoch)


			log_dict_metrics = jax.tree_util.tree_map(reshape_utils.unravel_dict, log_dict["metrics"])
			if("Losses" in log_dict.keys()):
				loss_dict = {f"losses/{key}": log_dict["Losses"][key] for key in log_dict["Losses"]}
				for key in loss_dict.keys():
					if key not in wandb_log_dict:
						wandb_log_dict[key] = []

					wandb_log_dict[key].append(loss_dict[key])

			energy_dict = {f"energies/{key}": log_dict["energies"][key] for key in log_dict["energies"]}

			batch_log_dict = self.__calculate_reporting(energy_graph_batch.graph,
				log_dict_metrics["energies"], gt_normed_energies, log_dict_metrics["spin_log_probs"], log_dict_metrics["free_energies"])
			batch_log_dict["solution_prob_mean"] = log_dict_metrics["solution_prob_mean"]
			batch_log_dict["solution_prob_min"] = log_dict_metrics["solution_prob_min"]

			for key in batch_log_dict.keys():
				if key not in wandb_log_dict:
					wandb_log_dict[key] = []

				wandb_log_dict[key].append(batch_log_dict[key][None, ...])

			for key in energy_dict.keys():
				if key not in wandb_log_dict:
					wandb_log_dict[key] = []

				if key.startswith("energies/energy"):
					wandb_log_dict[key + "_min"] = []
					wandb_log_dict[key + "_max"] = []
					wandb_log_dict[key + "_std"] = []
					wandb_log_dict[key + "_min"].append(energy_dict[key].min())
					wandb_log_dict[key + "_max"].append(energy_dict[key].max())
					wandb_log_dict[key + "_std"].append(energy_dict[key].std())
				
				wandb_log_dict[key].append(energy_dict[key])

			save_metrics_at_epoch["eval/energy"].append(batch_log_dict["mean_energy"])
			save_metrics_at_epoch["eval/gt_energy"].append(batch_log_dict["mean_gt_energy"])
			save_metrics_at_epoch["eval/rel_error"].append(batch_log_dict["rel_error"])

		for key in save_metrics_at_epoch.keys():
			self.save_metrics_dict[key].append(np.mean(np.concatenate(save_metrics_at_epoch[key], axis = 0)))

		self.__plot_figures(log_dict, mode=mode, step=epoch)
		eval_log_dict = {
			f"{mode}/epoch": epoch,
			f"{mode}/epochs_since_best": self.epochs_since_best,
			f"{mode}/best_rel_error": self.best_rel_error,
			f"{mode}/best_energy": self.best_energy,
		}

		for key in wandb_log_dict.keys():
			try:
				res = np.concatenate(np.concatenate(wandb_log_dict[key], axis=1), axis=0)
				eval_log_dict[f"{mode}/" + key] = np.mean(res)
			except:

				eval_log_dict[f"{mode}/" + key] = np.mean(wandb_log_dict[key])

		average_energy = eval_log_dict[f"{mode}/mean_energy"]
		mean_rel_energies = eval_log_dict[f"{mode}/rel_error"]

		if mean_rel_energies < self.best_rel_error:
			self.best_rel_error = mean_rel_energies
			self.epochs_since_best = 0
		else:
			self.epochs_since_best += 1

		if average_energy < self.best_energy:
			self.__save_checkpoint(f"best_{self.wandb_run_id}.pickle", epoch=epoch, eval_dict=eval_log_dict)
			self.best_energy = average_energy
			self.epochs_since_best = 0
		else:
			self.epochs_since_best += 1

		if epoch > 0 and epoch % self.save_modulo == 0:
			self.__save_checkpoint(f"{self.wandb_run_id}_{epoch}.pickle", epoch=epoch)

		self.__save_checkpoint(f"{self.wandb_run_id}_last_epoch.pickle", epoch=epoch)

		if self.use_wandb:
			wandb.log(eval_log_dict, step=epoch)
		self.last_eval_log = eval_log_dict
		return eval_log_dict

	def test(self, mode = "test"):

		dataloader = self.dataloader_test

		wandb_log_dict = {}
		save_metrics_at_epoch = {}
		save_metrics_at_epoch["eval/energy"] = []
		save_metrics_at_epoch["eval/gt_energy"] = []
		save_metrics_at_epoch["eval/rel_error"] = []

		time_dict = {"forward_pass": [], "CE": []}
		energy_mat_list = []
		gt_energy_mat_list = []

		for iter, (batch_dict) in enumerate(dataloader):
			gt_normed_energies = batch_dict["energies"]
			print("batchsize is", len(gt_normed_energies))

			graph_batch, energy_graph_batch = self._prepare_graphs(batch_dict, mode = mode)

			self.key, subkey = jax.random.split(self.key)
			batched_key = jax.random.split(subkey, num = len(jax.devices()))

			loss, (log_dict, _) = self.TrainerClass.evaluation_step(self.params, graph_batch, energy_graph_batch, self.T, batched_key, mode = mode)

			time_dict["forward_pass"].append(log_dict["time"]["forward_pass"])
			time_dict["CE"].append(log_dict["time"]["CE"])


			log_dict_metrics = jax.tree_util.tree_map(reshape_utils.unravel_dict, log_dict["metrics"])
			if("Losses" in log_dict.keys()):
				loss_dict = {f"losses/{key}": log_dict["Losses"][key] for key in log_dict["Losses"]}
				for key in loss_dict.keys():
					if key not in wandb_log_dict:
						wandb_log_dict[key] = []

					wandb_log_dict[key].append(loss_dict[key])

			### TODO fix this logging so that batchsize does not have an effect anymore
			energy_dict = {f"energies/{key}": log_dict["energies"][key] for key in log_dict["energies"]}

			batch_log_dict = self.__calculate_reporting(energy_graph_batch.graph,
				log_dict_metrics["energies"], gt_normed_energies, log_dict_metrics["spin_log_probs"], log_dict_metrics["free_energies"])

			batch_CE_log_dict = self.__calculate_reporting(energy_graph_batch.graph,
				log_dict_metrics["energies_CE"], gt_normed_energies, log_dict_metrics["spin_log_probs"], log_dict_metrics["free_energies"], prefix= "CE")

			energy_mat_list.append(log_dict_metrics["energies_CE"])
			gt_energy_mat_list.append(gt_normed_energies)

			for key in batch_log_dict.keys():
				if key not in wandb_log_dict:
					wandb_log_dict[key] = []

				wandb_log_dict[key].append(batch_log_dict[key][None, ...])

			for key in batch_CE_log_dict.keys():
				if key not in wandb_log_dict:
					wandb_log_dict[key] = []

				wandb_log_dict[key].append(batch_CE_log_dict[key][None, ...])

			for key in energy_dict.keys():
				if key not in wandb_log_dict:
					wandb_log_dict[key] = []

				wandb_log_dict[key].append(energy_dict[key])

		CE_time = np.sum(time_dict["CE"])
		forw_pass_time = np.sum(time_dict["forward_pass"])
		overall_time = CE_time + forw_pass_time
		eval_log_dict = {
			f"{mode}/epochs_since_best": self.epochs_since_best,
			f"{mode}/best_rel_error": self.best_rel_error,
			f"{mode}/best_energy": self.best_energy,
			f"{mode}/CE_time": CE_time,
			f"{mode}/forward_pass_time": forw_pass_time,
			f"{mode}/overall_time": overall_time,
			f"{mode}/energy_mat": np.concatenate(energy_mat_list, axis = 0),
			f"{mode}/gt_energy_mat": np.concatenate(gt_energy_mat_list, axis = 0)
		}

		for key in wandb_log_dict.keys():
			try:
				if(key == "best_energy"):
					print(key)
				res = np.concatenate(np.concatenate(wandb_log_dict[key], axis=1), axis=0)
				eval_log_dict[f"{mode}/" + key] = np.mean(res)
				eval_log_dict[f"{mode}/" + "std" + key] = np.std(res)
			except:
				eval_log_dict[f"{mode}/" + key] = np.mean(wandb_log_dict[key])
				eval_log_dict[f"{mode}/" + "std" + key] = np.std(wandb_log_dict[key])

		self.__save_test_dict(eval_log_dict, self.TrainerClass.eval_step_factor)

		return eval_log_dict


	def _prepare_graphs(self, batch_dict,  mode = "train"):
		if(self.graph_mode != "Transformer"):
			if(self.graph_mode == "U_net"):
				input_graph_dict = pmap_batch_U_net_graph_dict_and_pad(batch_dict["U_net_graph_dict"], k = self.pad_k)
				_, energy_graph = self._pad_graphs(batch_dict["input_graph"], batch_dict["energy_graph"])
				#self.pad_k = 1.3
			elif(self.graph_mode != "U_net"):

				input_graph, energy_graph = self._pad_graphs(batch_dict["input_graph"], batch_dict["energy_graph"], mode = mode)

				input_graph_dict = {"graphs": [input_graph]}
		elif(self.graph_mode == "Transformer"):
			input_graph, energy_graph = self._pad_graphs(batch_dict["input_graph"], batch_dict["energy_graph"])

			# X_pos_encoding = self._add_node_encoding(input_graph)
			# input_graph = input_graph._replace(nodes = X_pos_encoding)
			input_graph_dict = {"graphs": [input_graph]}

		return input_graph_dict, energy_graph

	def _pad_graphs(self, input_graph, energy_graph, mode = "train"):
		if(self.graph_mode != "Transformer"):
			if (True):
				#input_graph = pad_graph_to_nearest_power_of_k(input_graph, k=self.pad_k)
				#energy_graph = pad_graph_to_nearest_power_of_k(energy_graph, k=self.pad_k)
				if (mode == "train"):
					dataset_statistics_dict_input = {"grid_num": self.grid_num,  "edge_grid_factor": self.edge_grid_factor,
											   "min_nodes": self.dataloader_train.smallest_n_nodes_input_graph,
											   "min_edges": self.dataloader_train.smallest_n_edges_input_graph,
												 "max_nodes": self.dataloader_train.largest_n_nodes_input_graph,
												 "max_edges": self.dataloader_train.largest_n_edges_input_graph
													 }
					dataset_statistics_dict_energy = {"grid_num": self.grid_num,  "edge_grid_factor": self.edge_grid_factor,
											   "min_nodes": self.dataloader_train.smallest_n_nodes_energy_graph,
											   "min_edges": self.dataloader_train.smallest_n_edges_energy_graph,
											  "max_nodes": self.dataloader_train.largest_n_nodes_energy_graph,
											  "max_edges": self.dataloader_train.largest_n_edges_energy_graph
													  }
				elif (mode == "eval" or mode == "val"):
					dataset_statistics_dict_input = {"grid_num": self.grid_num,  "edge_grid_factor": self.edge_grid_factor,
											   "min_nodes": self.dataloader_val.smallest_n_nodes_input_graph,
											   "min_edges": self.dataloader_val.smallest_n_edges_input_graph,
											 "max_nodes": self.dataloader_val.largest_n_nodes_input_graph,
											 "max_edges": self.dataloader_val.largest_n_edges_input_graph
													 }
					dataset_statistics_dict_energy = {"grid_num": self.grid_num,  "edge_grid_factor": self.edge_grid_factor,
											   "min_nodes": self.dataloader_val.smallest_n_nodes_energy_graph,
											   "min_edges": self.dataloader_val.smallest_n_edges_energy_graph,
											  "max_nodes": self.dataloader_val.largest_n_nodes_energy_graph,
											  "max_edges": self.dataloader_val.largest_n_edges_energy_graph
													  }
				else:
					dataset_statistics_dict_input = {"grid_num": self.grid_num,  "edge_grid_factor": self.edge_grid_factor,
											   "min_nodes": self.dataloader_test.smallest_n_nodes_input_graph,
											   "min_edges": self.dataloader_test.smallest_n_edges_input_graph,
													 "max_nodes": self.dataloader_test.largest_n_nodes_input_graph,
													 "max_edges": self.dataloader_test.largest_n_edges_input_graph
													 }
					dataset_statistics_dict_energy = {"grid_num": self.grid_num,  "edge_grid_factor": self.edge_grid_factor,
											   "min_nodes": self.dataloader_test.smallest_n_nodes_energy_graph,
											   "min_edges": self.dataloader_test.smallest_n_edges_energy_graph,
													  "max_nodes": self.dataloader_test.largest_n_nodes_energy_graph,
													  "max_edges": self.dataloader_test.largest_n_edges_energy_graph
													  }

				input_graph = jraph_utils.pmap_graph_list_better(input_graph, dataset_statistics_dict_input)
				energy_graph = jraph_utils.pmap_graph_list_better(energy_graph, dataset_statistics_dict_energy)
				# print("here energy graph", energy_graph.nodes.shape, energy_graph.n_node, energy_graph.n_edge, energy_graph.edges.shape)
				# print("here", input_graph.nodes.shape, input_graph.n_node, input_graph.n_edge, input_graph.edges.shape)
				# raise ValueError("")
				# input_graph = jraph_utils.pmap_graph_list(input_graph, k = self.pad_k)
				# energy_graph = jraph_utils.pmap_graph_list(energy_graph, k = self.pad_k)

			else:
				input_graph = jraph_utils.pmap_graph_list(input_graph, k=self.pad_k)
				energy_graph = jraph_utils.pmap_graph_list(energy_graph, k=self.pad_k)
		else:
			input_graph = jraph_utils.pmap_transformer_list(input_graph, k=self.pad_k)
			energy_graph = jraph_utils.pmap_transformer_list(energy_graph, k=self.pad_k)


		return input_graph, energy_graph

	def __plot_figures(self, log_dict, mode = "eval", step = None):
		if not self.use_wandb:
			return
		if "figures" in log_dict.keys():
			plt_dict = {}
			figure_dict = log_dict["figures"]

			for figure_key in figure_dict:
				if "type" in figure_dict[figure_key]:
					fig_type = figure_dict[figure_key]["type"]
					if(fig_type == "Ising"):
						fig = plt.figure()

						if figure_key == "Ising_states":
							fig, axs = plt.subplots(2, 2, figsize=(10, 10))

							# Plot each image in the grid
							for i in range(4):
								X_0 = figure_dict[figure_key]["X_0"][i]
								row = i // 2
								col = i % 2
								axs[row, col].matshow(X_0)
								axs[row, col].set_title(f"{figure_key} {i + 1}")

							# Adjust layout to prevent overlap
							plt.tight_layout()

						else:
							print(figure_key, figure_dict[figure_key].keys())
							plt.title(figure_key)
							est_free_energies = np.asarray(figure_dict[figure_key]["y_axis"]).flatten()
							n_states = np.asarray(figure_dict[figure_key]["x_axis"]).flatten()
							plt.plot(n_states, est_free_energies, marker = "x")
							if("baseline" in figure_dict[figure_key].keys()):
								plt.axhline(y=figure_dict[figure_key]["baseline"], color='r', linestyle='-')
							plt.ylabel("Estimate")
							plt.xlabel("Number of Basis States")

						plt_dict[f"{mode}/figures/{figure_key}"] = wandb.Image(fig)
						plt.close("all")
				else:
					fig = plt.figure()
					plt.title(figure_key)

					y_values = np.mean(figure_dict[figure_key]["y_values"],axis = 0)
					x_values = np.arange(0, len(y_values))

					plt.plot(x_values, y_values, "-x")

					plt_dict[f"{mode}/figures/{figure_key}"] = wandb.Image(fig)
					plt.close("all")

			if step is not None:
				wandb.log(plt_dict, step=step, commit=False)
			else:
				wandb.log(plt_dict)

	@partial(jax.jit, static_argnums=(0,))
	def calc_mean_prob(self,graphs, spin_log_probs):
		### TODO implement this for more than oe device
		graphs = jax.tree_util.tree_map(lambda x: jnp.concatenate(x, axis = 0), graphs)
		nodes = graphs.nodes
		n_node = graphs.n_node
		n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
		graph_idx = jnp.arange(n_graph)
		total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
		node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_num_nodes)
		mask_not_person = jnp.where(graphs.globals["node_types"] < 3, 1, 0)[..., None, None]
		mean_prob_per_graph = jraph.segment_sum(jnp.exp(spin_log_probs) * mask_not_person, node_graph_idx, n_graph) / jraph.segment_sum(mask_not_person, node_graph_idx, n_graph)
		return mean_prob_per_graph[:-1]

	def __calculate_reporting(self, graphs, normed_energies, gt_normed_energies, spin_log_probs, normed_free_energies=np.nan, prefix = ""):
		gt_normed_energies = np.array(gt_normed_energies)
		if not np.isnan(normed_free_energies).all():
			mean_normed_free_energy = np.mean(normed_free_energies)
		else:
			mean_normed_free_energy = np.nan

		energies = np.array(normed_energies * self.std_energy + self.mean_energy)
		gt_energies = np.array(gt_normed_energies * self.std_energy + self.mean_energy)
		gt_energies = np.expand_dims(gt_energies, axis=-1)
		gt_energies = gt_energies[:energies.shape[0]]

		min_energies = np.min(energies, axis=1)

		# print("energies", min_energies.shape, energies.shape, gt_energies.shape)
		# print(min_energies - np.max(energies, axis=1))
		rel_error_matrix = np.abs(gt_energies - np.squeeze(energies, axis=-1)) / np.abs(gt_energies)
		best_rel_error_per_graph = np.abs(gt_energies - min_energies) / np.abs(gt_energies)

		mean_normed_energy = normed_energies
		mean_gt_energy = gt_energies
		mean_best_energy = min_energies
		mean_best_rel_error = best_rel_error_per_graph
		mean_best_10_percent_energy = np.mean(np.sort(energies, axis=1)[:, :max(1, int(0.1 * energies.shape[1]))], axis=1)
		#print(rel_error, mean_best_rel_error)


		mean_prob_per_graph = self.calc_mean_prob(graphs, spin_log_probs)

		if(np.isnan(np.mean(energies))):
			print(energies)
			raise ValueError("energies is nan")

		log_dict = {"mean_energy": energies, "mean_normed_energy": mean_normed_energy,
					"mean_gt_energy": mean_gt_energy, "mean_best_energy":  mean_best_energy, "rel_error": rel_error_matrix,
					"mean_best_rel_error": mean_best_rel_error, "mean_prob": mean_prob_per_graph, "mean_best_10_percent_energy": mean_best_10_percent_energy,}

		if self.problem_name == 'MVC':
			APR_per_graph = np.squeeze(energies, axis=-1) / gt_energies
			best_APR_per_graph = min_energies / gt_energies

			log_dict["APR"] = APR_per_graph
			log_dict["best_APR"] = best_APR_per_graph
		if self.problem_name == 'MaxCut':
			energies = np.squeeze(energies, axis = -1)
			num_edges = 2*np.reshape(graphs.n_edge[:,:-1],(graphs.n_edge.shape[0]*(graphs.n_edge.shape[1]-1),1))
			MaxCut_Value = num_edges/4 - energies/2
			best_MaxCut_Value = np.max(num_edges/4 - energies/2, axis = -1)
			gt_MaxCut_Value = num_edges/4 - gt_normed_energies/2

			log_dict["MaxCut_Value"] = MaxCut_Value
			log_dict["best_MaxCut_Value"] = best_MaxCut_Value
		else:
			APR = None
			best_APR = None

		if(prefix != ""):
			new_log_dict = {}
			for key in log_dict:
				new_log_dict[key + f"_{prefix}"] = log_dict[key]
			return new_log_dict

		return log_dict

	def __checkpoint_payload(self, epoch: int, eval_dict: dict = None):
		dict_to_save = {"params": self.params,
						"opt_state": self.opt_state,
						"T": self.T,
						"epoch": epoch,
						"config": self.config,
						"logs": self.save_metrics_dict
						}
		if eval_dict is not None:
			dict_to_save["eval_dict"] = eval_dict
		return dict_to_save

	def __save_checkpoint(self, file_name: str, epoch: int, eval_dict: dict = None):
		if not self.save_checkpoints:
			return
		path_folder = f"{self.path_to_models}/{self.wandb_run_id}/"

		if not os.path.exists(path_folder):
			os.makedirs(path_folder)

		with open(os.path.join(path_folder, file_name), 'wb') as f:
			pickle.dump(self.__checkpoint_payload(epoch=epoch, eval_dict=eval_dict), f)

	def __save_checkpoint_with_prefix(self, epoch: int, file_prefix: str, every_n_epochs: int = 100, eval_dict: dict = None):
		if not self.save_checkpoints:
			return
		if every_n_epochs <= 0 or epoch % every_n_epochs != 0:
			return

		file_name = f"{file_prefix}_{self.wandb_run_id}_epoch_{epoch}.pickle"
		self.__save_checkpoint(file_name, epoch=epoch, eval_dict=eval_dict)

	def __save_test_dict(self, test_dict, eval_step_factor):
		if not self.save_checkpoints:
			return
		path_folder = f"{self.path_to_models}/{self.wandb_old_run_id}/"

		if not os.path.exists(path_folder):
			os.makedirs(path_folder)

		file_name = f"{self.wandb_old_run_id}_test_dict_eval_step_factor_{eval_step_factor}_{self.TrainerClass.N_test_basis_states}.pickle"

		with open(os.path.join(path_folder, file_name), 'wb') as f:
			pickle.dump(test_dict, f)

	def __save_stuff(self, stuff_dict, stuff_name = ""):
		if not self.save_checkpoints:
			return
		path_folder = f"{self.path_to_models}/{self.wandb_old_run_id}/"

		if not os.path.exists(path_folder):
			os.makedirs(path_folder)

		file_name = f"{self.wandb_old_run_id}_stuff_dict_{stuff_name}.pickle"

		with open(os.path.join(path_folder, file_name), 'wb') as f:
			pickle.dump(stuff_dict, f)
