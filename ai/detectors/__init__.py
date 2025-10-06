
"""
AI Detectors Package
Provides multi-camera detection capabilities for drone inspection
"""

from .my_detection import (
    MultiCameraController,
    SharedModelManager,
    CameraWorker,
    GPUManager,
    gpu_manager
)

__all__ = [
    'MultiCameraController',
    'SharedModelManager', 
    'CameraWorker',
    'GPUManager',
    'gpu_manager'
]

__version__ = '1.0.0'