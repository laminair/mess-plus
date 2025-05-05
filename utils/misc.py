import logging
import random
import numpy as np
import torch
import os


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler(f'mess_plus.log'),
        logging.StreamHandler()
    ]
)


def is_nested_list(obj):
    return isinstance(obj, list) and any(isinstance(item, list) for item in obj)


def set_all_seeds(seed=42):
    """
    Set random seeds for reproducibility across PyTorch, NumPy, Python's random,
    Pandas, and CUDA if available.

    Args:
        seed (int): The seed value to use for all random number generators.
                    Default is 42.
    """

    # Set Python's random seed
    random.seed(seed)

    # Set NumPy's random seed
    np.random.seed(seed)

    # Set PyTorch's random seeds for both CPU and CUDA
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Extra settings for CUDA determinism
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set environment variable for CUDA operations
    os.environ['PYTHONHASHSEED'] = str(seed)

    logger.info(f"All random seeds have been set to {seed}.")
