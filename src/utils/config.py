"""Configuration management for experiments.

Loads YAML configuration files and provides a clean interface
for accessing experiment parameters.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Dictionary containing the configuration.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save a configuration dictionary to a YAML file.
    
    Args:
        config: Configuration dictionary to save.
        output_path: Path to save the YAML file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def generate_experiment_id(config: Dict[str, Any]) -> str:
    """Generate a unique experiment ID from configuration.
    
    Format: {model}_{dataset}_SEED{seed}
    
    Args:
        config: Experiment configuration dictionary.
        
    Returns:
        Experiment ID string.
    """
    model = config.get('experiment', {}).get('name', 'unknown')
    dataset = config.get('data', {}).get('dataset', 'unknown')
    seed = config.get('experiment', {}).get('seed', 0)
    
    # Add sequence length for LSTM models
    seq_len = config.get('data', {}).get('sequence_length', None)
    
    parts = [model.upper(), dataset]
    if seq_len is not None:
        parts.append(f"SEQ{seq_len}")
    parts.append(f"SEED{seed}")
    
    return '_'.join(parts)


def update_config(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge override values into a base configuration.
    
    Args:
        base: Base configuration dictionary.
        overrides: Override values to merge.
        
    Returns:
        Merged configuration dictionary.
    """
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = update_config(result[key], value)
        else:
            result[key] = value
    return result
