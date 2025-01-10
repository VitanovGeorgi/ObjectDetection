"""
General utility functions.
"""

import os
import numpy as np
import random
import torch
import torch.distributed as dist

import yaml
from omegaconf import OmegaConf, DictConfig



def init_distributed_mode(args):
    # Initialize the distributed process group
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(args.local_rank)
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()


def load_config(path) -> dict:
    """
    Load a yaml config file.

    Args:
        path (str): Path to the yaml file.

    Returns:
        dict: The loaded config file.
    """

    with open(path, 'r') as file:
        cfg = yaml.safe_load(file)

    return cfg

def load_config_omega(path) -> DictConfig:
    """
    Load a yaml config file.

    Args:
        path (str): Path to the yaml file.

    Returns:
        DictConfig: The loaded config file.
    """

    
    cfg = OmegaConf.load(path)

    return cfg

def reset_random_seeds(seed) -> torch.Generator:
    """
    Reset the random seeds for reproducibility.

    Args:
        seed (int): The seed to use.
    """
    # Let me know if I'm missing something here :) - MV
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    gen = torch.manual_seed(seed)

    return gen