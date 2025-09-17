# utils/__init__.py
"""
Utilities package
Helper functions and utility classes
"""

from .coordinates import (
    degrees_to_radians, radians_to_degrees, normalize_angle,
    calculate_distance_2d, calculate_distance_3d, calculate_bearing,
    rotate_point_2d, CoordinateTransformer
)
from .file_utils import (
    FileManager, file_manager, save_application_settings, 
    load_application_settings, update_recent_files
)

__all__ = [
    'degrees_to_radians',
    'radians_to_degrees', 
    'normalize_angle',
    'calculate_distance_2d',
    'calculate_distance_3d',
    'calculate_bearing',
    'rotate_point_2d',
    'CoordinateTransformer',
    'FileManager',
    'file_manager',
    'save_application_settings',
    'load_application_settings', 
    'update_recent_files'
]