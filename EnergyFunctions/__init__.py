
from .HCPEnergy import HCPEnergyClass

noise_distribution_registry = { "HCP": HCPEnergyClass}



def get_Energy_class(config):

    noise_distr_str = config["problem_name"]

    if(noise_distr_str in noise_distribution_registry.keys()):
        Energy_class = noise_distribution_registry[noise_distr_str]
    else:
        raise ValueError(f"CO Problem {noise_distr_str} is not implemented")

    return Energy_class(config)