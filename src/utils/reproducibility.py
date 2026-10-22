"""
Reproducibility utilities — seed setting, environment recording.
"""
import json
import os
import platform
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and (if available) PyTorch/TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_environment_info() -> Dict[str, Any]:
    """Collect environment metadata for reproducibility tracking."""
    env = {
        "timestamp": datetime.utcnow().isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }

    # Library versions
    libraries = [
        "numpy", "pandas", "scikit-learn", "xgboost", "torch",
        "tensorflow", "statsmodels", "geopandas", "shapely",
        "optuna", "shap", "fastapi", "streamlit",
    ]
    versions: Dict[str, str] = {}
    for lib in libraries:
        try:
            import importlib
            mod = importlib.import_module(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not installed"
    env["library_versions"] = versions

    return env


def save_environment(path: str | Path) -> None:
    """Save environment info as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = get_environment_info()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
