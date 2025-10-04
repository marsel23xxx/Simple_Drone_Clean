# config/settings.py

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"
UI_ASSETS = ASSETS_DIR / "images"
CV_CUSTOM = PROJECT_ROOT / "cv_custom"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# Application configuration
APP_CONFIG = {
    'app_name': 'Drone Control Center',
    'app_version': '2.0.0',
    'organization': 'DroneControl Solutions',
    'window_title': 'Professional Drone Ground Control Station',
    'window_size': (1920, 1080),
    'min_window_size': (1200, 800)
}

# Network configuration
NETWORK_CONFIG = {
    'tcp_listen_port': 12345,
    'websocket_ip': '192.168.1.88',
    'websocket_port': 9000,
    'drone_data_port': 8889,
    'connection_timeout': 5.0,
    'reconnect_interval': 3.0
}

# Point cloud configuration
POINTCLOUD_CONFIG = {
    'process_frame_skip': 4,
    'display_frame_skip': 1,
    'max_points_render': 100000,
    'point_size': 3,
    'default_max_height': 1.5,
    'color_gradient': {
        'low_color': [1.0, 0.0, 0.0],   # Red
        'high_color': [0.0, 0.0, 1.0]   # Blue
    }
}

# UI Configuration
UI_CONFIG = {
    'update_intervals': {
        'status_update': 1000,      # ms
        'drone_data': 100,          # ms
        'fps_calculation': 1000     # ms
    },
    'animation_duration': 200,       # ms
    'grid_size': 1.0,
    'zoom_limits': (0.5, 500.0),
    'default_zoom': 20.0
}

# File paths
FILE_PATHS = {
    'coordinates_file': DATA_DIR / 'coordinates.json',
    'settings_file': DATA_DIR / 'app_settings.json',
    'logs_dir': DATA_DIR / 'logs'
}

# Asset paths
ASSET_PATHS = {
    'logo': UI_ASSETS / 'LOGO R BG-012.png',
    'drone_display': UI_ASSETS / 'drone-display.png',
    'compass': UI_ASSETS / 'compas.png',
    'emergency': UI_ASSETS / 'emergency.png',
    'altitude': UI_ASSETS / 'altitude.png',
    'drone_top': UI_ASSETS / 'Drone 2.png',
    'drone_bottom': UI_ASSETS / 'Drone 3.png',
    'drone_png': UI_ASSETS / 'DRONETOP.png'
}

# Ensure log directory exists
FILE_PATHS['logs_dir'].mkdir(exist_ok=True)

# Debug configuration
DEBUG_CONFIG = {
    'enable_logging': True,
    'log_level': 'INFO',
    'console_output': True,
    'file_output': True,
    'max_log_files': 10
}

# Drone command configuration
COMMAND_CONFIG = {
    'available_commands': [
        'hover', 'stop', 'home', 'start', 
        'goto', 'coordinates', 'emergency'
    ],
    'command_timeout': 2.0,
    'retry_attempts': 3
}