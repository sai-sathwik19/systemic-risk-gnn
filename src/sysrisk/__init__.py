"""sysrisk - GNNs for systemic financial risk (Eisenberg-Noe clearing benchmark)."""
from .clearing import ClearingResult, clearing_fda, picard_dense, picard_sparse
from .dataset import generate_dataset, load_or_generate, network_to_data, summarize
from .generators import make_config, sample_network

__all__ = [
    "ClearingResult", "clearing_fda", "picard_dense", "picard_sparse",
    "generate_dataset", "load_or_generate", "network_to_data", "summarize",
    "make_config", "sample_network",
]
__version__ = "0.1.0"
