# config/__init__.py
"""
Configuration package for Drone Control Center
Provides centralized configuration management
"""

from .settings import (
    APP_CONFIG, NETWORK_CONFIG, POINTCLOUD_CONFIG, CV_CUSTOM,
    UI_CONFIG, FILE_PATHS, ASSET_PATHS, DEBUG_CONFIG, COMMAND_CONFIG
)

__version__ = "2.0.0"
__author__ = "Drone Control Solutions"