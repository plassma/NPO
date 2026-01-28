
from .HCPEnergy import HCPEnergyClass
from .CountdownEnergy import CountdownEnergyClass

energy_class_registry = { "HCP": HCPEnergyClass, "Countdown": CountdownEnergyClass}



def get_Energy_class(config):

    noise_distr_str = config["problem_name"]

    if(noise_distr_str in energy_class_registry.keys()):
        Energy_class = energy_class_registry[noise_distr_str]
    else:
        raise ValueError(f"CO Problem {noise_distr_str} is not implemented")

    return Energy_class(config)