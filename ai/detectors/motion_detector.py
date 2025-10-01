# =====================================================================
# FILE: motion_detector.py
# =====================================================================
from dataclasses import dataclass

from typing import Tuple
import os
import cv2
from ..gpu_manager import gpu_manager, img_processor

@dataclass
class MotionData:
    plate_center: Tuple[int, int]
    plate_radius: float
    marker_center: Tuple[int, int]
    current_angle: float
    total_rotation: float
    rotation_speed: float
    direction: str
    confidence: float = 0.9


class MotionDetector:
    def __init__(self, output_dir='ai_motion_captures'):
        self.output_dir = output_dir
        self.device = gpu_manager.device
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.motion_detection_mode = True
        self.angle_buffer = []
        self.previous_angle = None
        self.total_rotation = 0.0
        
        self.save_counter = 0
        
        print(f"Motion Detector initialized - Output: {self.output_dir}")
    
    def process_frame(self, frame):
        return {
            'motion_detected': False,
            'motion_data': None
        }
    
    def annotate_frame(self, frame, motion_results):
        annotated = frame.copy()
        
        if motion_results and motion_results.get('motion_detected'):
            cv2.putText(annotated, "MOTION DETECTED", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return annotated
    
    def get_capture_count(self) -> int:
        return self.save_counter


# Export semua classes
__all__ = ['RustDetector', 'CorrosionData', 'PositionData',
           'HazmatDetector', 'QRDetector', 'MotionDetector', 'MotionData']
