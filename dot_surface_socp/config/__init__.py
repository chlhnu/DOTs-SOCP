"""Configuration module for dot-surface-socp.

This module loads configuration from TOML files and provides constants
and utilities for the rest of the application.
"""

import os
import tomli
import logging


_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_LOGGING_CONFIG_PATH = os.path.join(_CONFIG_DIR, 'logging_config.toml')
_PATH_CONFIG_PATH = os.path.join(_CONFIG_DIR, 'path_config.toml')

# Load configuration from TOML files
def _load_config_file(config_path):
    try:
        with open(config_path, 'rb') as f:
            return tomli.load(f)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return {}

# =======================================
# Configuration of Paths
# =======================================
_path_config = _load_config_file(_PATH_CONFIG_PATH)
if 'paths' not in _path_config:
    raise ValueError("Missing 'paths' configuration item")

PATHS = _path_config['paths']

# =======================================
# Configuration of Logging Levels
# =======================================
_logging_config = _load_config_file(_LOGGING_CONFIG_PATH)

if 'log_levels' not in _logging_config:
    raise ValueError("Missing 'log_levels' configuration item")

log_levels = _logging_config['log_levels']
required_levels = ['debug', 'info', 'kkt', 'scaling']

for level in required_levels:
    if level not in log_levels:
        raise ValueError(f"Missing '{level}' log level configuration")

LOG_LEVELS = {
    'debug': log_levels['debug'],
    'info': log_levels['info'],
    'kkt': log_levels['kkt'],
    'scaling': log_levels['scaling']
}

# Register custom log levels with the logging module
logging.addLevelName(LOG_LEVELS['kkt'], 'KKT')
logging.addLevelName(LOG_LEVELS['scaling'], 'SCALING')