import random

import numpy as np
import torch


def set_seed(seed: int = 6304) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_generator(seed: int = 6304) -> torch.Generator:
    """Create a seeded PyTorch DataLoader generator."""

    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python inside a DataLoader worker."""

    worker_seed = torch.initial_seed() % (2 ** 32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)
