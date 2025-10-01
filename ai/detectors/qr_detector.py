"""
QR Code Detection Module
GPU-optimized QR detection with position analysis
"""

import cv2
import numpy as np
import math
import time
import datetime
import os
import hashlib
import re
from typing import Optional, List, Tuple
from ..gpu_manager import gpu_manager


class QRDetector:
    def __init__(self, output_dir='ai_qr_captures', qr_real_size=0.05, camera_height=1.5):
        self.output_dir = output_dir
        self.qr_real_size = qr_real_size
        self.camera_height = camera_height
        
        # QR Code detector
        self.qr_detector = cv2.QRCodeDetector()
        
        # GPU setup
        self.device = gpu_manager.device
        
        # Create folders
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "analytics"), exist_ok=True)
        
        # Camera parameters
        self.focal_length = 800
        self.frame_width = 640
        self.frame_height = 480
        self.frame_center_x = self.frame_width // 2
        self.frame_center_y = self.frame_height // 2
        
        # Detection tracking
        self.detection_count = {}
        self.saved_qr_codes = set()
        self.STABILIZATION_FRAMES = 3
        
        print(f"[QR] Detector initialized with GPU: {gpu_manager.cuda_available}")
    
    def update_frame_dimensions(self, width, height):
        """Update frame dimensions"""
        self.frame_width = width
        self.frame_height = height
        self.frame_center_x = width // 2
        self.frame_center_y = height // 2
    
    def calculate_xyz_position(self, qr_points, frame_shape):
        """Calculate XYZ position"""
        try:
            qr_points_np = np.array(qr_points, dtype=np.float32)
            
            rect = self.order_points(qr_points_np)
            
            side_vectors = np.array([
                rect[1] - rect[0],
                rect[3] - rect[0]
            ])
            side_lengths = np.linalg.norm(side_vectors, axis=1)
            qr_size_pixels = np.mean(side_lengths)
            
            center = np.mean(qr_points_np, axis=0)
            center_x, center_y = center
            
            frame_center = np.array([frame_shape[1] / 2, frame_shape[0] / 2])
            
            pixel_offset = center - frame_center
            angle_per_pixel = np.array([60 / frame_shape[1], 45 / frame_shape[0]])
            angles = pixel_offset * angle_per_pixel
            angle_y, angle_x = angles
            angle_x = -angle_x
            
            x_distance = (self.qr_real_size * self.focal_length) / qr_size_pixels
            
            y_distance = x_distance * math.tan(math.radians(angle_y))
            z_vertical_offset = x_distance * math.tan(math.radians(angle_x))
            z_distance = self.camera_height - z_vertical_offset
            
            if abs(angle_y) < 5:
                direction = "CENTER"
            elif angle_y > 0:
                direction = "RIGHT"
            else:
                direction = "LEFT"
            
            return {
                'x_distance': x_distance,
                'y_distance': y_distance,
                'z_distance': z_distance,
                'angle_x': angle_x,
                'angle_y': angle_y,
                'qr_size_pixels': qr_size_pixels,
                'center_x': center_x,
                'center_y': center_y,
                'direction': direction
            }
            
        except Exception as e:
            print(f"[QR] XYZ calculation error: {e}")
            return None
    
    def order_points(self, pts):
        """Order points"""
        s = np.sum(pts, axis=1)
        d = pts[:, 0] - pts[:, 1]
        
        rect = np.zeros((4, 2), dtype=np.float32)
        rect[0] = pts[np.argmin(s)]
        rect[1] = pts[np.argmax(d)]
        rect[2] = pts[np.argmax(s)]
        rect[3] = pts[np.argmin(d)]
        
        return rect
    
    def create_object_id(self, qr_data, xyz_info):
        """Create unique object ID"""
        center_x = int(xyz_info['center_x'])
        center_y = int(xyz_info['center_y'])
        qr_hash = hashlib.md5(qr_data.encode('utf-8')).hexdigest()[:8]
        return f"qr_{qr_hash}_{center_x//50}_{center_y//50}"
    
    def process_frame(self, frame: np.ndarray) -> List[dict]:
        """Process frame for QR detection"""
        self.update_frame_dimensions(frame.shape[1], frame.shape[0])
        
        # Detect QR codes
        data, bbox, _ = self.qr_detector.detectAndDecode(frame)
        
        detections = []
        
        if data and bbox is not None:
            if len(bbox.shape) == 3:
                for i, points in enumerate(bbox):
                    if data:
                        qr_points = points.astype(np.float32)
                        xyz_info = self.calculate_xyz_position(qr_points, frame.shape)
                        
                        if xyz_info:
                            detection_data = {
                                'qr_data': data,
                                'xyz_info': xyz_info,
                                'qr_points': qr_points,
                                'timestamp': datetime.datetime.now()
                            }
                            detections.append(detection_data)
            else:
                if data:
                    qr_points = bbox.astype(np.float32)
                    xyz_info = self.calculate_xyz_position(qr_points, frame.shape)
                    
                    if xyz_info:
                        detection_data = {
                            'qr_data': data,
                            'xyz_info': xyz_info,
                            'qr_points': qr_points,
                            'timestamp': datetime.datetime.now()
                        }
                        detections.append(detection_data)
        
        return detections
    
    def annotate_frame(self, frame, qr_detections):
        """Annotate frame with QR detections"""
        annotated = frame.copy()
        
        if not qr_detections:
            cv2.putText(annotated, "SCANNING FOR QR CODES", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return annotated
        
        for detection in qr_detections:
            qr_data = detection['qr_data']
            xyz_info = detection['xyz_info']
            qr_points = detection['qr_points']
            
            # Draw boundary
            pts = qr_points.astype(int).reshape(-1, 1, 2)
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
            
            # Draw center
            center = (int(xyz_info['center_x']), int(xyz_info['center_y']))
            cv2.circle(annotated, center, 5, (0, 255, 0), -1)
            
            # Display info
            text_y = 30
            cv2.putText(annotated, f"QR: {qr_data[:20]}...", 
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            text_y += 25
            cv2.putText(annotated, f"X: {xyz_info['x_distance']:.3f}m", 
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            text_y += 20
            cv2.putText(annotated, f"Y: {xyz_info['y_distance']:.3f}m ({xyz_info['direction']})", 
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            text_y += 20
            cv2.putText(annotated, f"Z: {xyz_info['z_distance']:.3f}m", 
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated