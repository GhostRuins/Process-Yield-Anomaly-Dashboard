# config.py - Configuration management for production deployment

import os
from pathlib import Path
from typing import Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "model": {
        "default_contamination": 0.05,
        "default_n_estimators": 200,
        "random_state": 42,
        "test_size": 0.2,
        "cv_folds": 5,
    },
    "analysis": {
        "default_threshold": 3.0,
        "min_samples_for_stats": 5,
        "p_value_threshold": 0.05,
        "effect_size_threshold": 0.5,
    },
    "data": {
        "min_records": 10,
        "max_records": 1000000,
        "allowed_extensions": [".csv"],
    },
    "storage": {
        "model_dir": "models",
        "cache_dir": "cache",
        "logs_dir": "logs",
    },
    "ui": {
        "default_theme": "light",
        "page_title": "Advanced Process Yield Anomaly Detector",
    }
}

class Config:
    """Configuration manager with file persistence."""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.config = DEFAULT_CONFIG.copy()
        self.load()
        self._ensure_directories()
    
    def load(self):
        """Load configuration from file or use defaults."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    self._deep_update(self.config, file_config)
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}. Using defaults.")
        else:
            logger.info("No config file found. Using defaults.")
    
    def save(self):
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'model.default_contamination')."""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
    
    def _deep_update(self, base: Dict, updates: Dict):
        """Recursively update nested dictionaries."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        for dir_type in ["model_dir", "cache_dir", "logs_dir"]:
            dir_path = Path(self.get(f"storage.{dir_type}"))
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {dir_path}")

# Global config instance
config = Config()

