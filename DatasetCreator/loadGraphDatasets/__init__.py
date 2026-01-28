"""
Dataset Generator
"""

from .CountdownDatasetGenerator import CountdownDatasetGenerator
from .HCPDatasetGenerator import HCPDatasetGenerator

dataset_generator_registry = {"HCP_dummy": HCPDatasetGenerator, "Countdown_small": CountdownDatasetGenerator,}


def get_dataset_generator(config):
	"""
	:param config: config dictionary specifying the dataset that should be generated
	:return: dataset generator class
	"""
	dataset_name = config["dataset_name"]

	for dataset in dataset_generator_registry.keys():
		if dataset in dataset_name:
			dataset_generator = dataset_generator_registry[dataset]
			return dataset_generator(config)
	raise ValueError(f"Dataset {dataset_name} is not implemented")



