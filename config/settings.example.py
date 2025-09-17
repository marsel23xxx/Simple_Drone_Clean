"""
Example configuration file for Drone Control Center
Copy this to settings.py and modify according to your setup
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"
UI_ASSETS = ASSETS_DIR / "images"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# Application configuration
APP_CONFIG = {
    'app_name': 'Drone Control Center',
    'app_version': '2.0.0',
    'organization': 'Your Organization Name',
    'window_title': 'Professional Drone Ground Control Station',
    'window_size': (1920, 1080),
    'min_window_size': (1200, 800)
}

# Network configuration - MODIFY THESE FOR YOUR SETUP
NETWORK_CONFIG = {
    # TCP port for receiving point cloud data
    'tcp_listen_port': 12345,
    
    # Drone WebSocket connection details
    'websocket_ip': '192.168.1.88',    # Change to your drone's IP
    'websocket_port': 9000,
    
    # UDP port for drone telemetry data
    'drone_data_port': 8889,
    
    # Connection timeouts
    'connection_timeout': 5.0,
    'reconnect_interval': 3.0
}

# Point cloud visualization configuration
POINTCLOUD_CONFIG = {
    # Frame processing settings
    'process_frame_skip': 4,        # Process every Nth frame (higher = better performance)
    'display_frame_skip': 1,        # Display every Nth processed frame
    
    # Rendering limits
    'max_points_render': 100000,    # Maximum points to render (adjust for performance)
    'point_size': 3,                # Size of rendered points
    
    # Height filtering
    'default_max_height': 1.5,      # Default height filter in meters
    
    # Color gradient for height visualization
    'color_gradient': {
        'low_color': [1.0, 0.0, 0.0],   # Red for low points
        'high_color': [0.0, 0.0, 1.0]   # Blue for high points
    }
}

# User interface configuration
UI_CONFIG = {
    # Update intervals in milliseconds
    'update_intervals': {
        'status_update': 1000,      # General status updates
        'drone_data': 100,          # Drone telemetry updates  
        'fps_calculation': 1000     # FPS calculation frequency
    },
    
    # Animation and interaction
    'animation_duration': 200,       # Animation duration in ms
    'grid_size': 1.0,               # Grid size in meters
    'zoom_limits': (0.5, 500.0),   # Min and max zoom levels
    'default_zoom': 20.0            # Initial zoom level
}

# File paths configuration
FILE_PATHS = {
    'coordinates_file': DATA_DIR / 'coordinates.json',
    'settings_file': DATA_DIR / 'app_settings.json',
    'logs_dir': DATA_DIR / 'logs'
}

# Asset file paths - UPDATE THESE IF YOU MOVE ASSETS
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

# Debug and logging configuration
DEBUG_CONFIG = {
    'enable_logging': True,
    'log_level': 'INFO',            # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'console_output': True,         # Print logs to console
    'file_output': True,            # Write logs to file
    'max_log_files': 10            # Maximum number of log files to keep
}

# Drone command configuration
COMMAND_CONFIG = {
    # Available drone commands
    'available_commands': [
        'hover',        # Hover in place
        'stop',         # Emergency stop
        'home',         # Return to home
        'start',        # Start mission
        'goto',         # Go to specific coordinates
        'coordinates',  # Set waypoint coordinates
        'emergency'     # Emergency protocol
    ],
    
    # Command timeouts and retries
    'command_timeout': 2.0,         # Timeout for command responses
    'retry_attempts': 3             # Number of retry attempts
}

# Hardware-specific optimizations
HARDWARE_CONFIG = {
    # Set to True if you have a dedicated GPU
    'use_gpu_acceleration': False,
    
    # Reduce these values for slower hardware
    'max_concurrent_threads': 4,
    'memory_limit_mb': 1024,
    
    # Display optimizations
    'enable_vsync': True,
    'target_fps': 60
}

# Drone-specific settings
DRONE_CONFIG = {
    # Drone type and capabilities
    'drone_type': 'generic',        # generic, dji, ardupilot, px4
    
    # Flight limits and safety
    'max_altitude': 120.0,          # Maximum altitude in meters
    'max_distance': 500.0,          # Maximum distance from home in meters
    'min_battery_level': 20,        # Minimum battery level for flight
    
    # Navigation settings
    'waypoint_tolerance': 0.5,      # Waypoint arrival tolerance in meters
    'max_velocity': 5.0,            # Maximum velocity in m/s
    'max_acceleration': 2.0,        # Maximum acceleration in m/s²
    
    # Emergency settings
    'auto_land_battery': 10,        # Auto-land when battery below this %
    'return_home_signal_loss': True # RTH on signal loss
}

# Environment-specific settings
ENVIRONMENT_CONFIG = {
    # Operating environment
    'indoor_mode': False,           # Indoor vs outdoor operation
    'gps_available': True,          # GPS availability
    
    # Weather limits
    'max_wind_speed': 10.0,         # m/s
    'min_visibility': 1000.0,       # meters
    
    # Coordinate system
    'coordinate_system': 'NED',     # NED (North-East-Down) or ENU (East-North-Up)
    'reference_frame': 'local'      # local, global, utm
}

# Performance monitoring
MONITORING_CONFIG = {
    'enable_performance_monitoring': True,
    'log_frame_times': False,
    'log_memory_usage': False,
    'performance_alert_threshold': 100,  # ms
}

# Advanced settings - modify only if you know what you're doing
ADVANCED_CONFIG = {
    # Threading settings
    'tcp_thread_priority': 'normal',
    'websocket_thread_priority': 'normal',
    'ui_thread_priority': 'high',
    
    # Memory management
    'garbage_collection_interval': 60,  # seconds
    'memory_cleanup_threshold': 80,     # percentage
    
    # Network buffer sizes
    'tcp_buffer_size': 8192,
    'websocket_buffer_size': 4096,
    
    # Point cloud processing
    'use_numpy_optimizations': True,
    'enable_point_culling': True,
    'spatial_hash_grid_size': 0.1
}

# Export/Import settings
EXPORT_CONFIG = {
    'default_export_format': 'json',
    'include_metadata': True,
    'compress_exports': False,
    'export_directory': DATA_DIR / 'exports'
}

# Integration settings
INTEGRATION_CONFIG = {
    # External system integration
    'mavlink_enabled': False,
    'ros_enabled': False,
    'custom_protocol_enabled': False,
    
    # API settings
    'rest_api_enabled': False,
    'rest_api_port': 8080,
    'api_authentication': False
}