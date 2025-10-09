# ai/detectors/my_detection.py

import sys
import os
import cv2
import numpy as np
import threading
import time
import logging
import datetime
import math
import re
import torch
import torchvision.transforms as T
import torchvision.models.segmentation as models
from PIL import Image
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from collections import deque
from functools import lru_cache
from core.drone_parser import DroneParser
import gc

# Try to import easyocr
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[WARNING] EasyOCR not available. Landolt Ring OCR will be disabled.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ID_PATTERN = re.compile(r"^[0-9]+[A-Z]?$")

# ============================================================================
# CAMERA ANALYTICS CLASS (dari landolt.py)
# ============================================================================

class CameraAnalytics:
    def __init__(self, frame_width=640, frame_height=480):
        """Inisialisasi camera analytics untuk mengkonversi 2D ke 3D coordinates"""
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_center_x = frame_width // 2
        self.frame_center_y = frame_height // 2
        
        self.focal_length_pixel = 800
        self.sensor_width_mm = 5.6
        self.pixel_size_mm = self.sensor_width_mm / frame_width
        self.reference_landolt_diameter_mm = 20
        
        self.initial_surface_z = 0
        self.camera_height_offset = 0
        self.reference_distance_m = 0.5
        
        self.is_calibrated = False
        self.calibration_samples = []
        self.max_calibration_samples = 10
        
    def calibrate_reference(self, ring_radius_pixels, known_distance_m=0.5):
        if ring_radius_pixels > 0:
            ring_diameter_pixels = ring_radius_pixels * 2
            self.focal_length_pixel = (ring_diameter_pixels * known_distance_m * 1000) / self.reference_landolt_diameter_mm
            self.calibration_samples.append(self.focal_length_pixel)
            
            if len(self.calibration_samples) >= 5:
                self.focal_length_pixel = np.mean(self.calibration_samples)
                self.is_calibrated = True
                return True
        return False
    
    def pixels_to_3d_coordinates(self, x_pixel, y_pixel, ring_radius_pixels):
        if ring_radius_pixels <= 0:
            return 0, 0, 0, "unknown"
        
        ring_diameter_pixels = ring_radius_pixels * 2
        distance_m = (self.reference_landolt_diameter_mm * self.focal_length_pixel) / (ring_diameter_pixels * 1000)
        x_distance = distance_m
        
        dx_from_center = x_pixel - self.frame_center_x
        y_lateral = (dx_from_center * distance_m) / self.focal_length_pixel
        
        if abs(y_lateral) < 0.02:
            direction = "center"
        elif y_lateral < 0:
            direction = "left"
        else:
            direction = "right"
        
        dy_from_center = self.frame_center_y - y_pixel
        z_height = (dy_from_center * distance_m) / self.focal_length_pixel + self.camera_height_offset
        
        return x_distance, y_lateral, z_height, direction
    
    def reset_reference(self):
        self.initial_surface_z = 0
        self.camera_height_offset = 0
        self.is_calibrated = False
        self.calibration_samples = []

# ============================================================================
# QR DETECTOR 3D CLASS (dari qrcodeupdate.py)
# ============================================================================

class QRDetector3D:
    def __init__(self, camera_id, output_base, qr_real_size=0.05, camera_height=1.0):
        """Enhanced QR Detector with 3D analysis and auto-save"""
        self.camera_id = camera_id
        self.output_base = output_base
        self.qr_real_size = qr_real_size
        self.camera_height = camera_height
        
        # Create directories
        os.makedirs(f"{output_base}/images", exist_ok=True)
        os.makedirs(f"{output_base}/analytics", exist_ok=True)
        
        # QR Detector
        self.qr_detector = cv2.QRCodeDetector()
        
        # Camera parameters
        self.focal_length = 800
        self.frame_width = 640
        self.frame_height = 480
        self.frame_center_x = self.frame_width // 2
        self.frame_center_y = self.frame_height // 2
        
        # Detection tracking
        self.qr_position_history = {}
        self.last_direction_data = {}
        self.saved_qr_codes = set()
        self.qr_save_counter = 0
        
        # Auto-save parameters
        self.stabilization_frames = 5
        self.qr_detection_count = {}
        self.direction_change_threshold = 15.0
        
        print(f"[CAM {camera_id}] QR Detector initialized - Output: {output_base}")
    
    def _sanitize_filename(self, text, max_length=20):
        """Convert text to safe filename"""
        if not text:
            return "empty"
        safe_text = re.sub(r'[^\w\-_.]', '_', str(text))
        safe_text = re.sub(r'_+', '_', safe_text)
        safe_text = safe_text.strip('_')
        if len(safe_text) > max_length:
            safe_text = safe_text[:max_length]
        if not safe_text:
            safe_text = "sanitized"
        return safe_text
    
    def _generate_qr_hash(self, qr_data):
        """Generate short hash for QR data"""
        import hashlib
        hash_object = hashlib.md5(qr_data.encode('utf-8'))
        return hash_object.hexdigest()[:8]
    
    def _get_direction_text(self, angle_y):
        """Convert Y angle to direction text"""
        if abs(angle_y) < 5:
            return "CENTER"
        elif angle_y > 0:
            return "RIGHT"
        else:
            return "LEFT"
    
    def calculate_xyz_position(self, qr_points, frame_shape):
        """Calculate XYZ position and angles from QR code"""
        try:
            # Calculate QR size in pixels
            rect = self._order_points(qr_points)
            width_pixels = np.linalg.norm(rect[1] - rect[0])
            height_pixels = np.linalg.norm(rect[3] - rect[0])
            qr_size_pixels = (width_pixels + height_pixels) / 2
            
            # Calculate center
            center_x = np.mean(qr_points[:, 0])
            center_y = np.mean(qr_points[:, 1])
            
            # Calculate angles
            pixel_offset_x = center_x - self.frame_center_x
            angle_per_pixel_x = 60 / frame_shape[1]
            angle_y = pixel_offset_x * angle_per_pixel_x
            
            pixel_offset_y = center_y - self.frame_center_y
            angle_per_pixel_y = 45 / frame_shape[0]
            angle_x = -pixel_offset_y * angle_per_pixel_y
            
            # Calculate distances
            x_distance = (self.qr_real_size * self.focal_length) / qr_size_pixels
            y_distance = x_distance * math.tan(math.radians(angle_y))
            z_vertical_offset = x_distance * math.tan(math.radians(angle_x))
            z_distance = self.camera_height - z_vertical_offset
            
            return {
                'x_distance': x_distance,
                'y_distance': y_distance,
                'z_distance': z_distance,
                'angle_x': angle_x,
                'angle_y': angle_y,
                'qr_size_pixels': qr_size_pixels,
                'center_x': center_x,
                'center_y': center_y,
                'camera_height': self.camera_height,
                'vertical_offset': z_vertical_offset
            }
        except Exception as e:
            logger.error(f"XYZ calculation error: {e}")
            return None
    
    def _order_points(self, pts):
        """Order points for perspective transform"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        d = pts[:, 0] - pts[:, 1]
        rect[0] = pts[np.argmin(s)]
        rect[1] = pts[np.argmax(d)]
        rect[2] = pts[np.argmax(s)]
        rect[3] = pts[np.argmin(d)]
        return rect
    
    def _check_direction_change(self, qr_data, xyz_info):
        """Check if camera direction changed significantly"""
        try:
            current_angle_y = xyz_info['angle_y']
            current_direction = self._get_direction_text(current_angle_y)
            
            if qr_data not in self.qr_position_history:
                self.qr_position_history[qr_data] = []
                self.last_direction_data[qr_data] = {
                    'angle_y': current_angle_y,
                    'direction': current_direction
                }
                return False
            
            last_data = self.last_direction_data[qr_data]
            angle_change = abs(current_angle_y - last_data['angle_y'])
            direction_changed = current_direction != last_data['direction']
            
            self.qr_position_history[qr_data].append({
                'angle_y': current_angle_y,
                'direction': current_direction,
                'x_distance': xyz_info['x_distance']
            })
            
            if len(self.qr_position_history[qr_data]) > 10:
                self.qr_position_history[qr_data].pop(0)
            
            if angle_change > self.direction_change_threshold or direction_changed:
                self.last_direction_data[qr_data] = {
                    'angle_y': current_angle_y,
                    'direction': current_direction
                }
                return True
            
            return False
        except Exception as e:
            logger.error(f"Direction check error: {e}")
            return False
    
    def _create_qr_object_id(self, qr_data, xyz_info):
        """Create unique ID for QR detection"""
        direction = self._get_direction_text(xyz_info['angle_y'])
        distance_zone = int(xyz_info['x_distance'] * 10)
        return f"{qr_data}_{direction}_{distance_zone}"
    
    def _save_qr_detection(self, frame, qr_data, xyz_info, event_type="DETECTION"):
        """Save QR detection with analytics"""
        try:
            self.qr_save_counter += 1
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            qr_hash = self._generate_qr_hash(qr_data)
            safe_preview = self._sanitize_filename(qr_data, 10)
            direction = self._get_direction_text(xyz_info['angle_y'])
            
            # Save image
            img_filename = f"qr_cam{self.camera_id}_{event_type}_{self.qr_save_counter}_{timestamp}_{safe_preview}_{qr_hash}.jpg"
            img_path = os.path.join(self.output_base, "images", img_filename)
            cv2.imwrite(img_path, frame)
            
            # Save analytics
            analytics_filename = f"qr_cam{self.camera_id}_{event_type}_{self.qr_save_counter}_{timestamp}_{safe_preview}_{qr_hash}.txt"
            analytics_path = os.path.join(self.output_base, "analytics", analytics_filename)
            
            with open(analytics_path, 'w', encoding='utf-8') as f:
                f.write(f"=== QR CODE DETECTION ANALYSIS ===\n")
                f.write(f"Camera: {self.camera_id}\n")
                f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Event Type: {event_type}\n")
                f.write(f"Save Counter: {self.qr_save_counter}\n")
                f.write(f"QR Hash: {qr_hash}\n")
                f.write(f"{'='*60}\n\n")
                
                f.write(f"QR DATA:\n")
                f.write(f"{qr_data}\n\n")
                
                f.write(f"3D POSITION ANALYSIS:\n")
                f.write(f"X - Distance from Camera: {xyz_info['x_distance']:.3f}m\n")
                f.write(f"Y - Horizontal Position: {xyz_info['y_distance']:+.3f}m ({direction})\n")
                f.write(f"Z - Height from Ground: {xyz_info['z_distance']:.3f}m\n")
                f.write(f"Camera Height: {xyz_info['camera_height']:.3f}m\n\n")
                
                f.write(f"ANGLES:\n")
                f.write(f"Horizontal Angle (Y): {xyz_info['angle_y']:+.1f}°\n")
                f.write(f"Vertical Angle (X): {xyz_info['angle_x']:+.1f}°\n")
                f.write(f"Direction: {direction}\n\n")
                
                f.write(f"DETECTION DETAILS:\n")
                f.write(f"QR Size (pixels): {xyz_info['qr_size_pixels']:.1f}px\n")
                f.write(f"Center Position: ({xyz_info['center_x']:.0f}, {xyz_info['center_y']:.0f})px\n\n")
                
                if qr_data in self.qr_position_history and len(self.qr_position_history[qr_data]) > 1:
                    f.write(f"POSITION HISTORY:\n")
                    for i, hist in enumerate(self.qr_position_history[qr_data][-5:]):
                        f.write(f"  {i+1}. Dir:{hist['direction']}, Angle:{hist['angle_y']:.1f}°, Dist:{hist['x_distance']:.2f}m\n")
            
            print(f"\n[CAM {self.camera_id}] QR SAVED #{self.qr_save_counter} - {event_type}")
            print(f"  Image: {img_filename}")
            print(f"  Analytics: {analytics_filename}")
            print(f"  3D: X={xyz_info['x_distance']:.3f}m, Y={xyz_info['y_distance']:+.3f}m, Z={xyz_info['z_distance']:.3f}m")
            print(f"  Direction: {direction}")
            
            return True
        except Exception as e:
            logger.error(f"[CAM {self.camera_id}] QR save error: {e}")
            return False
    
    def process_qr_enhanced(self, frame):
        """Enhanced QR detection with auto-save"""
        try:
            data, bbox, _ = self.qr_detector.detectAndDecode(frame)
            detections = []
            
            if data and bbox is not None:
                if len(bbox.shape) == 3:
                    points_list = bbox
                else:
                    points_list = [bbox]
                
                for points in points_list:
                    if data and cv2.contourArea(points) > 1000:
                        qr_points = points.astype(np.float32)
                        
                        # Calculate 3D position
                        xyz_info = self.calculate_xyz_position(qr_points, frame.shape)
                        
                        if xyz_info:
                            # Create object ID
                            object_id = self._create_qr_object_id(data, xyz_info)
                            
                            # Track detection
                            if object_id not in self.qr_detection_count:
                                self.qr_detection_count[object_id] = 0
                            self.qr_detection_count[object_id] += 1
                            
                            # Check direction change
                            direction_changed = self._check_direction_change(data, xyz_info)
                            
                            # Auto-save logic
                            should_save = False
                            event_type = "DETECTION"
                            
                            if direction_changed and object_id not in self.saved_qr_codes:
                                should_save = True
                                event_type = "DIRECTION_CHANGE"
                            elif (self.qr_detection_count[object_id] >= self.stabilization_frames and 
                                  object_id not in self.saved_qr_codes):
                                should_save = True
                                event_type = "STABLE"
                            
                            if should_save:
                                self._save_qr_detection(frame, data, xyz_info, event_type)
                                self.saved_qr_codes.add(object_id)
                            
                            detection = {
                                'qr_data': data,
                                'qr_points': qr_points,
                                'xyz_info': xyz_info,
                                'object_id': object_id,
                                'detection_count': self.qr_detection_count[object_id],
                                'direction_changed': direction_changed
                            }
                            detections.append(detection)
            
            # Clean up old detections
            current_ids = {d['object_id'] for d in detections}
            old_ids = set(self.qr_detection_count.keys()) - current_ids
            for old_id in old_ids:
                del self.qr_detection_count[old_id]
                self.saved_qr_codes.discard(old_id)
            
            return detections
        
        except Exception as e:
            logger.error(f"[CAM {self.camera_id}] QR processing error: {e}")
            return []

# ============================================================================
# MOTION DATA CLASS
# ============================================================================

class DotOnPlateTracker:
    def __init__(self, camera_id, output_base):
        """
        Dot-on-Plate tracker (Canny + inner-contour marker) with motion detection
        """
        self.camera_id = camera_id
        self.output_base = output_base
        
        # Create directories
        os.makedirs(f"{output_base}/images", exist_ok=True)
        os.makedirs(f"{output_base}/analytics", exist_ok=True)
        
        # Frame dimensions
        self.frame_width = 640
        self.frame_height = 480

        # --- Motion Detection State ---
        self.motion_detection_mode = True
        self.angle_buffer = []
        self.motion_buffer_size = 15
        self.motion_threshold_deg = 8.0
        self.motion_speed_threshold = 3.0
        self.stable_frames_to_stop = 45
        self.stable_frame_count = 0
        self.motion_confirmation_count = 0
        self.motion_confirmation_required = 5
        self.noise_filter_threshold = 0.5

        # --- Tracking state ---
        self.previous_angle = None
        self.total_rotation = 0.0
        self.rotation_history = []
        self.plate_center = None
        self.plate_radius = None

        # --- Canny/contour params ---
        self.min_area = 50
        self.min_circularity = 0.80
        self.canny_low = 0
        self.canny_high = 122
        self.blur_ksize = 7

        # --- Inner contour (marker) detection params ---
        self.inner_min_area = 5

        # --- Direction hysteresis + stability ---
        self.direction = "Stable"
        self.dir_cum_delta_deg = 0.0
        self.dir_threshold_deg = 5.0
        self.stable_speed_deg_s = 2.0
        self.per_frame_epsilon_deg = 0.3

        # --- Debug/terminal state ---
        self.is_tracking = False
        self.last_direction_print = None

        # --- Auto Save & Analytics ---
        self.auto_save_enabled = True
        self.save_counter = 0
        self.last_save_time = 0
        self.save_interval = 2.0
        
        # --- Camera & Position Parameters ---
        self.camera_height = 1.500
        self.assumed_plate_diameter = 0.200
        self.camera_focal_length_px = 500
        
        print(f"[CAM {camera_id}] Motion Tracker initialized - Output: {output_base}")

    def detect_motion(self, current_angle, current_time):
        """Detect if object is actually rotating based on angle history"""
        if len(self.angle_buffer) > 0:
            last_angle = self.angle_buffer[-1][0]
            angle_diff = current_angle - last_angle
            
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            elif angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            if abs(math.degrees(angle_diff)) < self.noise_filter_threshold:
                current_angle = last_angle
        
        self.angle_buffer.append((current_angle, current_time))
        
        if len(self.angle_buffer) > self.motion_buffer_size:
            self.angle_buffer.pop(0)
        
        if len(self.angle_buffer) < 8:
            self.motion_confirmation_count = 0
            return False
        
        first_angle = self.angle_buffer[0][0]
        last_angle = self.angle_buffer[-1][0]
        
        angle_diff = last_angle - first_angle
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        elif angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        total_rotation_deg = abs(math.degrees(angle_diff))
        
        time_diff = self.angle_buffer[-1][1] - self.angle_buffer[0][1]
        avg_speed = total_rotation_deg / time_diff if time_diff > 0 else 0
        
        recent_speed = 0
        if len(self.angle_buffer) >= 5:
            recent_first = self.angle_buffer[-5][0]
            recent_last = self.angle_buffer[-1][0]
            recent_diff = recent_last - recent_first
            if recent_diff > math.pi:
                recent_diff -= 2 * math.pi
            elif recent_diff < -math.pi:
                recent_diff += 2 * math.pi
            recent_rotation = abs(math.degrees(recent_diff))
            recent_time_diff = self.angle_buffer[-1][1] - self.angle_buffer[-5][1]
            recent_speed = recent_rotation / recent_time_diff if recent_time_diff > 0 else 0
        
        motion_detected = ((total_rotation_deg > self.motion_threshold_deg and 
                           avg_speed > self.motion_speed_threshold) or
                          recent_speed > (self.motion_speed_threshold * 2))
        
        if motion_detected:
            self.motion_confirmation_count += 1
        else:
            self.motion_confirmation_count = 0
        
        return self.motion_confirmation_count >= self.motion_confirmation_required

    def check_motion_stopped(self, rotation_speed):
        """Check if motion has stopped"""
        if abs(rotation_speed) < (self.stable_speed_deg_s * 0.5):
            self.stable_frame_count += 1
        else:
            self.stable_frame_count = 0
        
        return self.stable_frame_count >= self.stable_frames_to_stop

    def calculate_world_position(self, plate_center, plate_radius_px, dot_center):
        """Calculate real world position in meters"""
        if plate_center is None or dot_center is None:
            return None
            
        cx, cy = plate_center
        dx, dy = dot_center
        
        distance_to_plate = (self.assumed_plate_diameter * self.camera_focal_length_px) / (2 * plate_radius_px)
        
        center_offset_px = cx - (self.frame_width / 2)
        horizontal_distance = (center_offset_px * distance_to_plate) / self.camera_focal_length_px
        
        vertical_offset_px = (self.frame_height / 2) - cy
        vertical_angle = math.atan(vertical_offset_px / self.camera_focal_length_px)
        height_from_camera = distance_to_plate * math.tan(vertical_angle)
        height_from_ground = self.camera_height + height_from_camera
        
        horizontal_angle = math.degrees(math.atan(horizontal_distance / distance_to_plate))
        vertical_angle_deg = math.degrees(vertical_angle)
        
        return {
            'x_distance': distance_to_plate,
            'y_horizontal': horizontal_distance,
            'z_height': height_from_ground,
            'horizontal_angle': horizontal_angle,
            'vertical_angle': vertical_angle_deg,
            'plate_diameter': self.assumed_plate_diameter
        }

    def save_detection_data(self, frame, plate_info, dot_center, angle_rad, rotation_speed, world_pos):
        """Save screenshot with FULL VISUAL OVERLAY like the example image"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create annotated frame with ALL visual elements
        annotated_frame = frame.copy()
        
        # Draw plate circle (green)
        if plate_info:
            center = (plate_info[0], plate_info[1])
            radius = plate_info[2]
            cv2.circle(annotated_frame, center, radius, (0, 255, 0), 3)
            cv2.circle(annotated_frame, center, 3, (0, 255, 0), -1)
        
        # Draw marker dot (blue) and orientation arrow
        if dot_center and plate_info:
            # Blue dot at marker
            cv2.circle(annotated_frame, dot_center, 8, (255, 0, 0), -1)
            
            # Magenta line from center to marker
            cv2.arrowedLine(annotated_frame, center, dot_center, 
                           (255, 0, 255), 4, tipLength=0.3)
            
            # Yellow orientation arrow from marker
            arrow_length = 50
            end_x = int(dot_center[0] + arrow_length * math.cos(angle_rad))
            end_y = int(dot_center[1] + arrow_length * math.sin(angle_rad))
            cv2.arrowedLine(annotated_frame, dot_center, (end_x, end_y), 
                           (0, 255, 255), 3, tipLength=0.3)
        
        # Calculate linear speed
        estimated_radius = self.assumed_plate_diameter / 2
        linear_speed = math.radians(rotation_speed) * estimated_radius
        
        # Add BLACK INFO BOX with text overlay (bottom-left corner)
        info_lines = [
            f"Mode: TRACKING AKTIF",
            f"Current Angle: {math.degrees(angle_rad):.2f}°",
            f"Total Rotation: {math.degrees(self.total_rotation):.2f}°",
            f"Speed: {linear_speed:.1f} m/s",
            f"Direction: {self.direction}",
            f"Stable Count: {self.stable_frame_count}/{self.stable_frames_to_stop}"
        ]
        
        if world_pos:
            info_lines.extend([
                f"Distance: {world_pos['x_distance']:.3f}m",
                f"Y-Offset: {world_pos['y_horizontal']:+.3f}m",
                f"Height: {world_pos['z_height']:.3f}m"
            ])
        
        info_lines.append(f"Auto-Save: ON ({self.save_counter + 1})")
        
        # Draw black background box
        text_bg_height = len(info_lines) * 15 + 6
        panel_width = 250
        panel_y_start = self.frame_height - text_bg_height - 10
        cv2.rectangle(annotated_frame, (10, panel_y_start), 
                     (panel_width, self.frame_height - 10), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (10, panel_y_start), 
                     (panel_width, self.frame_height - 10), (255, 255, 255), 1)
        
        # Draw text lines
        for i, text in enumerate(info_lines):
            text_y = panel_y_start + 14 + i * 15
            cv2.putText(annotated_frame, text, (15, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Add green "Baterei_Motionon | Saved: X" text at top-left
        status_text = f"Baterei_Motion | Saved: {self.save_counter + 1}"
        cv2.putText(annotated_frame, status_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Save the annotated screenshot
        screenshot_path = os.path.join(self.output_base, "images", 
                                      f"motion_cam{self.camera_id}_{timestamp}_{self.save_counter:04d}.jpg")
        cv2.imwrite(screenshot_path, annotated_frame)
        
        # Save analytics text file
        analytics_path = os.path.join(self.output_base, "analytics",
                                     f"motion_cam{self.camera_id}_{timestamp}_{self.save_counter:04d}.txt")
        
        with open(analytics_path, 'w', encoding='utf-8') as f:
            f.write("DOT-ON-PLATE TRACKING ANALYSIS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Camera: {self.camera_id}\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Detection ID: {self.save_counter + 1:04d}\n\n")
            
            f.write("POSITION ANALYSIS (in METERS from ground level 0,0):\n")
            if world_pos:
                f.write(f"X - Distance from Camera: {world_pos['x_distance']:.3f} m\n")
                if world_pos['y_horizontal'] >= 0:
                    f.write(f"Y - Horizontal Distance: {world_pos['y_horizontal']:.3f} m (Right)\n")
                else:
                    f.write(f"Y - Horizontal Distance: {abs(world_pos['y_horizontal']):.3f} m (Left)\n")
                f.write(f"Z - Height from Ground: {world_pos['z_height']:.3f} m\n")
                f.write(f"Camera Height: {self.camera_height:.3f} m (above ground)\n")
                f.write(f"Detected Plate Diameter: {world_pos['plate_diameter']:.3f} m\n")
                f.write(f"Horizontal Angle: {world_pos['horizontal_angle']:.1f}°\n")
                f.write(f"Vertical Angle: {world_pos['vertical_angle']:.1f}°\n\n")
            
            f.write("ROTATION ANALYSIS:\n")
            f.write(f"Current Angle: {math.degrees(angle_rad):.1f}°\n")
            f.write(f"Total Rotation: {math.degrees(self.total_rotation):.1f}°\n")
            f.write(f"Rotation Speed: {linear_speed:.1f} m/s\n")
            f.write(f"Direction: {self.direction}\n\n")
            
            if plate_info:
                f.write("DETECTION DETAILS:\n")
                f.write(f"Plate Center (pixels): ({plate_info[0]}, {plate_info[1]})\n")
                f.write(f"Plate Radius (pixels): {plate_info[2]}\n")
            
            if dot_center:
                f.write(f"Marker Center (pixels): ({dot_center[0]}, {dot_center[1]})\n")
            
            f.write(f"Frame Size: {self.frame_width} x {self.frame_height}\n")
            f.write(f"Camera Focal Length (est.): {self.camera_focal_length_px} pixels\n")
        
        self.save_counter += 1
        print(f"\n[CAM {self.camera_id}] MOTION SAVED #{self.save_counter}")
        print(f"  Image: {screenshot_path}")
        print(f"  Analytics: {analytics_path}")

    # ... rest of the class methods remain the same (detect_plates_canny, detect_marker_in_plate, 
    # calculate_rotation_speed, process_frame, reset_tracking, annotate_frame)
    def detect_plates_canny(self, frame):
        """Detect circular plates using Canny edge detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 2)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            circularity = 4 * math.pi * area / (peri * peri)
            if circularity < self.min_circularity:
                continue
            score = circularity * area
            candidates.append((int(x), int(y), int(radius), score))
        candidates.sort(key=lambda t: t[3], reverse=True)
        return candidates

    def detect_marker_in_plate(self, frame, plate_info):
        """Detect marker inside the plate"""
        if plate_info is None:
            return None

        cx, cy, r = plate_info
        h, w = frame.shape[:2]

        # Extract ROI
        x0 = max(0, cx - r)
        y0 = max(0, cy - r)
        x1 = min(w, cx + r)
        y1 = min(h, cy + r)
        roi = frame[y0:y1, x0:x1]

        # Circular mask
        plate_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.circle(plate_mask, (cx - x0, cy - y0), max(1, r - 2), 255, -1)

        # Edges inside ROI
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        edges_in_plate = cv2.bitwise_and(edges, edges, mask=plate_mask)

        # Close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_in_plate = cv2.morphologyEx(edges_in_plate, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(edges_in_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        max_allowed_area = 0.5 * math.pi * (r ** 2)

        best_cnt = None
        best_area = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.inner_min_area or area > max_allowed_area:
                continue
            if area > best_area:
                best_area = area
                best_cnt = cnt

        if best_cnt is None:
            return None

        M = cv2.moments(best_cnt)
        if M["m00"] == 0:
            return None

        mx = int(M["m10"] / M["m00"]) + x0
        my = int(M["m01"] / M["m00"]) + y0
        return (mx, my)

    def calculate_rotation_speed(self, current_angle):
        """Calculate rotation speed in degrees per second"""
        if len(self.rotation_history) < 2:
            return 0.0
        time_diff = self.rotation_history[-1][1] - self.rotation_history[-2][1]
        angle_diff = current_angle - self.rotation_history[-2][0]
        # unwrap
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        elif angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        return math.degrees(angle_diff) / time_diff if time_diff > 0 else 0.0

    def process_frame(self, frame):
        """Process single frame and return tracking data"""
        current_time = time.time()

        # Detect circular plate candidates
        candidates = self.detect_plates_canny(frame)

        dot_center = None
        plate_info = None
        angle = 0.0
        rotation_speed = 0.0
        world_pos = None

        # Find plate with marker
        for (x, y, r, _score) in candidates:
            maybe_plate = (x, y, r)
            candidate_marker = self.detect_marker_in_plate(frame, maybe_plate)
            if candidate_marker is not None:
                plate_info = maybe_plate
                dot_center = candidate_marker
                break

        # Process based on current mode
        if plate_info is not None and dot_center is not None:
            cx, cy, r = plate_info
            px, py = dot_center
            angle = math.atan2(py - cy, px - cx)

            if self.motion_detection_mode:
                # Check for rotation
                motion_detected = self.detect_motion(angle, current_time)
                if motion_detected:
                    print(f"[CAM {self.camera_id}] Motion detected! Starting tracking...")
                    self.motion_detection_mode = False
                    self.is_tracking = True
                    self.previous_angle = angle
                    self.total_rotation = 0.0
                    self.rotation_history = [(angle, current_time)]
                    self.direction = "Stable"
                    self.dir_cum_delta_deg = 0.0
                    self.stable_frame_count = 0
                    self.tracking_start_time = current_time  # Mark when tracking started
            else:
                # Tracking mode
                world_pos = self.calculate_world_position((cx, cy), r, (px, py))

                if self.previous_angle is not None:
                    angle_diff = angle - self.previous_angle
                    if angle_diff > math.pi:
                        angle_diff -= 2 * math.pi
                    elif angle_diff < -math.pi:
                        angle_diff += 2 * math.pi
                    self.total_rotation += angle_diff

                    deg_diff = math.degrees(angle_diff)
                    if abs(deg_diff) < self.per_frame_epsilon_deg:
                        deg_diff = 0.0
                    self.dir_cum_delta_deg += deg_diff

                    self.rotation_history.append((angle, current_time))
                    if len(self.rotation_history) > 10:
                        self.rotation_history.pop(0)
                    rotation_speed = self.calculate_rotation_speed(angle)

                    if (abs(rotation_speed) < self.stable_speed_deg_s and
                            abs(self.dir_cum_delta_deg) < self.dir_threshold_deg):
                        self.direction = "Stable"
                    else:
                        if self.dir_cum_delta_deg >= self.dir_threshold_deg:
                            self.direction = "Clockwise"
                            self.dir_cum_delta_deg = 0.0
                        elif self.dir_cum_delta_deg <= -self.dir_threshold_deg:
                            self.direction = "CounterClockwise"
                            self.dir_cum_delta_deg = 0.0

                    # Check if motion stopped
                    if self.check_motion_stopped(rotation_speed):
                        print(f"[CAM {self.camera_id}] Motion stopped. Returning to motion detection mode...")
                        self.reset_tracking()

                else:
                    self.rotation_history.append((angle, current_time))
                    if len(self.rotation_history) > 10:
                        self.rotation_history.pop(0)

                self.previous_angle = angle

                # Auto-save
                if (self.auto_save_enabled and 
                    current_time - self.last_save_time > self.save_interval):
                    self.save_detection_data(frame, plate_info, dot_center, angle, 
                                           rotation_speed, world_pos)
                    self.last_save_time = current_time

        else:
            # No detection
            if not self.motion_detection_mode:
                self.stable_frame_count += 1
                if self.stable_frame_count >= self.stable_frames_to_stop:
                    print(f"[CAM {self.camera_id}] Lost detection. Returning to motion detection mode...")
                    self.reset_tracking()

        return {
            'plate_info': plate_info,
            'dot_center': dot_center,
            'angle': angle,
            'rotation_speed': rotation_speed,
            'world_pos': world_pos,
            'direction': self.direction,
            'total_rotation': self.total_rotation,
            'is_tracking': not self.motion_detection_mode
        }

    def reset_tracking(self):
        """Reset all tracking state"""
        self.motion_detection_mode = True
        self.is_tracking = False
        self.angle_buffer = []
        self.previous_angle = None
        self.total_rotation = 0.0
        self.rotation_history = []
        self.direction = "Stable"
        self.dir_cum_delta_deg = 0.0
        self.stable_frame_count = 0
        self.last_direction_print = None
        self.motion_confirmation_count = 0

    def annotate_frame(self, frame, tracking_data):
        """Draw tracking information on frame"""
        annotated = frame.copy()
        
        plate_info = tracking_data['plate_info']
        dot_center = tracking_data['dot_center']
        angle = tracking_data['angle']
        rotation_speed = tracking_data['rotation_speed']
        world_pos = tracking_data['world_pos']
        is_tracking = tracking_data['is_tracking']
        
        # Header
        if not is_tracking:
            header_text = "Menunggu Rotasi..."
            header_color = (0, 255, 255)
            cv2.putText(annotated, header_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, header_color, 2)
            
            # Minimal info panel
            info_text = [
                "Mode: MENUNGGU ROTASI",
                "Status: Tidak Ada Deteksi",
                "Mulai putar objek untuk tracking"
            ]
            
            text_bg_height = len(info_text) * 15 + 6
            panel_width = 250
            panel_y_start = self.frame_height - text_bg_height - 10
            cv2.rectangle(annotated, (10, panel_y_start), 
                         (panel_width, self.frame_height - 10), (0, 0, 0), -1)
            cv2.rectangle(annotated, (10, panel_y_start), 
                         (panel_width, self.frame_height - 10), (255, 255, 255), 1)
            for i, text in enumerate(info_text):
                text_y = panel_y_start + 14 + i * 15
                cv2.putText(annotated, text, (15, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            return annotated
        
        # Tracking mode - draw detection visuals
        header_text = "Deteksi Motion"
        header_color = (0, 255, 0)
        cv2.putText(annotated, header_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, header_color, 2)
        
        if plate_info is not None:
            center = (plate_info[0], plate_info[1])
            radius = plate_info[2]
            cv2.circle(annotated, center, radius, (0, 255, 0), 2)
            cv2.circle(annotated, center, 3, (0, 255, 0), -1)
            self.plate_center = center
            self.plate_radius = radius
            
        if dot_center is not None:
            cv2.circle(annotated, dot_center, 8, (255, 0, 0), -1)
            if self.plate_center is not None:
                cv2.arrowedLine(annotated, self.plate_center, dot_center, 
                               (255, 0, 255), 4, tipLength=0.3)
                
                # Orientation arrow
                arrow_length = 50
                end_x = int(dot_center[0] + arrow_length * math.cos(angle))
                end_y = int(dot_center[1] + arrow_length * math.sin(angle))
                cv2.arrowedLine(annotated, dot_center, (end_x, end_y), 
                               (0, 255, 255), 3, tipLength=0.3)

        # Info panel
        estimated_radius = self.assumed_plate_diameter / 2
        linear_speed = math.radians(rotation_speed) * estimated_radius
        
        info_text = [
            "Mode: TRACKING AKTIF",
            f"Current Angle: {math.degrees(angle):.1f}°",
            f"Total Rotation: {math.degrees(self.total_rotation):.1f}°", 
            f"Speed: {linear_speed:.1f} m/s",
            f"Direction: {self.direction}",
            f"Stable Count: {self.stable_frame_count}/{self.stable_frames_to_stop}"
        ]
        
        if world_pos:
            info_text.extend([
                f"Distance: {world_pos['x_distance']:.3f}m",
                f"Y-Offset: {world_pos['y_horizontal']:.3f}m",
                f"Height: {world_pos['z_height']:.3f}m"
            ])
        
        if self.auto_save_enabled:
            info_text.append(f"Auto-Save: ON ({self.save_counter})")
        
        text_bg_height = len(info_text) * 15 + 6
        panel_width = 250
        panel_y_start = self.frame_height - text_bg_height - 10
        cv2.rectangle(annotated, (10, panel_y_start), 
                     (panel_width, self.frame_height - 10), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, panel_y_start), 
                     (panel_width, self.frame_height - 10), (255, 255, 255), 1)
        for i, text in enumerate(info_text):
            text_y = panel_y_start + 14 + i * 15
            cv2.putText(annotated, text, (15, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        return annotated

# ============================================================================
# CRACK DATA CLASS
# ============================================================================

@dataclass
class CrackData:
    x: int
    y: int
    w: int
    h: int
    width: float
    height: float
    length: float
    max_thickness: float
    min_thickness: float
    endpoints: List[Tuple[int, int]]
    endpoints_world_coords: List[Tuple[float, float]]
    confidence: float = 0.0
    
    @property
    def classification(self) -> str:
        return ("Hairline" if self.max_thickness < 0.001 else
                "Fine" if self.max_thickness < 0.003 else
                "Medium" if self.max_thickness < 0.0075 else "Wide/Severe")
    
    @property
    def is_critical(self) -> bool:
        return self.max_thickness > 0.035 or self.length > 0.05

# ============================================================================
# RUST/CORROSION DATA CLASSES
# ============================================================================

@dataclass
class CorrosionData:
    severity_percentages: dict
    total_affected_percentage: float
    
    @property
    def dominant_class(self) -> str:
        class_names = {1: "Fair", 2: "Poor", 3: "Severe"}
        if not self.severity_percentages:
            return "Good"
        max_class = max(self.severity_percentages.items(), key=lambda x: x[1])
        return class_names.get(max_class[0], "Good")
    
    @property
    def is_critical(self) -> bool:
        severe_pct = self.severity_percentages.get(3, 0)
        poor_pct = self.severity_percentages.get(2, 0)
        return severe_pct > 20.0 or (poor_pct + severe_pct) > 50.0

@dataclass
class RustPositionData:
    x_distance_m: float
    y_distance_m: float
    z_height_ground_m: float
    square_size_m: float
    angle_horizontal: float
    angle_vertical: float
    
    def __str__(self):
        y_direction = "R" if self.y_distance_m > 0 else "L"
        return f"X: {self.x_distance_m:.2f}m, Y: {abs(self.y_distance_m):.2f}m{y_direction}, Z: {self.z_height_ground_m:.2f}m"

# ============================================================================
# GPU MANAGER
# ============================================================================

class GPUManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self.cuda_available = torch.cuda.is_available()
        self.gpu_memory_fraction = 0.25 if self.cuda_available else 0
        self.device = self._setup_gpu()
        self._setup_cuda_context()
        self._print_gpu_info()
        self._initialized = True
    
    def _setup_gpu(self):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            torch.cuda.empty_cache()
            return device
        return torch.device('cpu')
    
    def _setup_cuda_context(self):
        if self.cuda_available:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    
    def _print_gpu_info(self):
        if self.cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[GPU] Using: {gpu_name}")
            print(f"[GPU] Total Memory: {gpu_memory:.1f} GB")
        else:
            print("[GPU] CUDA not available - using CPU")
    
    def cleanup_memory(self):
        if self.cuda_available:
            torch.cuda.empty_cache()
            gc.collect()

gpu_manager = GPUManager()

# ============================================================================
# SHARED MODEL MANAGER
# ============================================================================

class SharedModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        print("\n[INIT] Loading shared models (once for all cameras)...")
        self.device = gpu_manager.device
        
        self.yolo_crack = None
        self.yolo_hazmat = None
        self.deeplabv3_rust = None
        self.easyocr_reader = None
        self.qr_detector = cv2.QRCodeDetector()
        
        self._initialized = True
        print("[OK] Shared model manager ready")
    
    def load_crack_model(self, weights='crack.pt'):
        if self.yolo_crack is None and os.path.exists(weights):
            self.yolo_crack = YOLO(weights)
            self.yolo_crack.fuse()
            print(f"[OK] Crack model loaded (shared)")
        return self.yolo_crack
    
    def load_hazmat_model(self, weights='hazmat.pt'):
        if self.yolo_hazmat is None and os.path.exists(weights):
            self.yolo_hazmat = YOLO(weights)
            print(f"[OK] Hazmat model loaded (shared)")
        return self.yolo_hazmat
    
    def load_rust_model(self, model_path='deeplabv3_corrosion_multiclass.pth'):
        if self.deeplabv3_rust is None and os.path.exists(model_path):
            model = models.deeplabv3_resnet50(weights=None, num_classes=4)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.deeplabv3_rust = model.eval().to(self.device)
            print(f"[OK] Rust model loaded (shared)")
        return self.deeplabv3_rust
    
    def load_ocr_reader(self):
        if self.easyocr_reader is None and EASYOCR_AVAILABLE:
            gpu_ok = gpu_manager.cuda_available
            self.easyocr_reader = easyocr.Reader(['en'], gpu=gpu_ok, verbose=False)
            print(f"[OK] EasyOCR loaded (shared) - GPU: {gpu_ok}")
        return self.easyocr_reader

# ============================================================================
# ENHANCED CAMERA WORKER WITH FULL RUST DETECTION
# ============================================================================

class CameraWorker:
    """Enhanced worker with Landolt, QR 3D, Hazmat, Crack, Rust, and FULL Motion detection"""
    
    def __init__(self, camera_id: int, shared_models: SharedModelManager, output_base='ai_captures'):
        self.camera_id = camera_id
        self.models = shared_models
        self.output_base = f"{output_base}_cam{camera_id}"
        self.device = gpu_manager.device
        self.drone_parser = DroneParser()
        self.drone_parser.start()
        
        # LANDOLT PARAMETERS
        self.dp = 1.0
        self.min_dist = 60
        self.param1 = 80
        self.param2 = 25
        self.min_radius = 25
        self.max_radius = 120
        
        # Camera Analytics
        self.camera_analytics = CameraAnalytics(frame_width=640, frame_height=480)
        
        # QR DETECTOR 3D
        qr_output = f"{self.output_base}/qr"
        self.qr_detector_3d = QRDetector3D(
            camera_id=camera_id,
            output_base=qr_output,
            qr_real_size=0.05,
            camera_height=1.0
        )
        
        # MOTION TRACKER - FULL IMPLEMENTATION
        motion_output = f"{self.output_base}/motion"
        self.motion_tracker = DotOnPlateTracker(
            camera_id=camera_id,
            output_base=motion_output
        )
        
        # Tracking parameters (Landolt)
        self.max_tracking_history = 5
        self.position_smoothing = 0.3
        self.confidence_threshold = 0.35
        self.tracking_distance_threshold = 40
        self.min_tracking_frames = 2
        
        # OCR parameters
        self.ocr_min_conf = 0.4
        self.ocr_expand_factor = 1.2
        self.ocr_cache_size = 100
        self._ocr_cache = {}
        self._ocr_cache_access = {}
        
        # Auto save parameters (Landolt)
        self.auto_save_enabled = True
        self.saved_positions = set()
        self.min_stable_frames = 3
        self.min_ring_confidence = 0.4
        self.position_tolerance = 30
        self.max_saves_per_session = 50
        self.save_counter = 0
        
        # Analytics file
        self.analytics_file = f"{self.output_base}/landolt/analytics.txt"
        self.analytics_update_interval = 30
        self.last_analytics_update = 0
        
        # HAZMAT PARAMETERS
        self.frame_width = 640
        self.frame_height = 480
        self.frame_center_x = self.frame_width // 2
        self.frame_center_y = self.frame_height // 2
        
        self.CAMERA_FOV_HORIZONTAL = 60
        self.CAMERA_FOV_VERTICAL = 45
        self.HAZMAT_CAMERA_HEIGHT = 1.5
        
        self.HAZMAT_CONFIDENCE_THRESHOLD = 0.65
        self.HAZMAT_STABILIZATION_FRAMES = 5
        self.hazmat_detection_count = {}
        self.hazmat_saved_objects = set()
        
        # CRACK DETECTION PARAMETERS
        self.WARP_SIZE = 300
        self.M_PER_PIXEL = 0.20 / self.WARP_SIZE
        self.CAMERA_HEIGHT = 1.500
        self.CAMERA_FOV_H = 60.0
        self.CAMERA_FOV_V = 45.0
        self.DETECTED_SQUARE_SIZE = 0.20
        
        # RUST DETECTION PARAMETERS
        self.RUST_WARP_SIZE = 300
        self.RUST_NUM_CLASSES = 4
        self.rust_class_colors = {1: (0, 255, 0), 2: (0, 255, 255), 3: (0, 0, 255)}
        
        self.RUST_CAMERA_FOCAL_LENGTH = 800
        self.RUST_REAL_SQUARE_SIZE = 0.20
        self.RUST_CAMERA_HEIGHT = 1.5
        
        self.rust_results_data = []
        self.rust_position_data = None
        self.rust_detection_count = {}
        self.rust_saved_objects = set()
        self.RUST_STABILIZATION_FRAMES = 5
        
        self.last_processed_frame = None
        
        # Hazard class mapping
        self.hazard_classes = {
            'explosive': 'Class 1 - Explosive',
            'flammable-gas': 'Class 2 - Flammable Gas',
            'non-flammable-gas': 'Class 2 - Non-Flammable Gas',
            'inhalation-hazard': 'Class 2 - Inhalation Hazard',
            'oxygen': 'Class 2 - Oxygen',
            'toxic-gas': 'Class 2 - Toxic Gas',
            'flammable-liquid': 'Class 3 - Flammable Liquid',
            'flammable-solid': 'Class 4 - Flammable Solid',
            'dangerous-when-wet': 'Class 4 - Dangerous when Wet',
            'spontaneously-combustible': 'Class 4 - Spontaneously Combustible',
            'oxidizer': 'Class 5 - Oxidizer',
            'organic-peroxide': 'Class 5 - Organic Peroxide',
            'toxic': 'Class 6 - Toxic',
            'poison': 'Class 6 - Poison',
            'pg-3': 'Class 6 - PG 3',
            'infectious-substance': 'Class 6 - Infectious',
            'radioactive': 'Class 7 - Radioactive',
            'fissile': 'Class 7 - Fissile',
            'corrosive': 'Class 8 - Corrosive'
        }
        
        # Create output directories
        os.makedirs(f"{self.output_base}/landolt/images", exist_ok=True)
        os.makedirs(f"{self.output_base}/landolt/data", exist_ok=True)
        os.makedirs(f"{self.output_base}/hazmat/images", exist_ok=True)
        os.makedirs(f"{self.output_base}/hazmat/analytics", exist_ok=True)
        os.makedirs(f"{self.output_base}/crack", exist_ok=True)
        os.makedirs(f"{self.output_base}/rust/images", exist_ok=True)
        os.makedirs(f"{self.output_base}/rust/analytics", exist_ok=True)
        
        # State
        self._landolt_tracking = []
        self.frame_count = 0
        self.current_landolt_results = []
        self.current_qr_detections = []
        self.current_hazmat_results = None
        self.current_crack_analysis = None
        self.current_rust_analysis = None
        self.current_motion_data = None
        self.crack_results_data = []
        
        self.hazmat_counter = 0
        self.qr_counter = 0
        self.landolt_counter = 0
        self.crack_counter = 0
        self.rust_counter = 0
        self.motion_counter = 0
        
        # Freeze state
        self.is_frozen = False
        self.frozen_frame = None
        
        print(f"[CAM {camera_id}] Enhanced Worker initialized - Output: {self.output_base}")
    
    # ========================================================================
    # LANDOLT DETECTION METHODS (keeping existing - not shown for brevity)
    # ========================================================================
    
    def _preprocess_for_ring(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        processed = cv2.GaussianBlur(bilateral, (3,3), 1.0)
        return processed
    
    def _check_ring_structure(self, contours, hierarchy, center, radius):
        if hierarchy is None or len(contours) < 2:
            return False, 0, 0, 0
        
        cx, cy = center
        best_outer = None
        best_score = 0
        
        for i, cnt in enumerate(contours):
            if hierarchy[0][i][3] == -1:
                area = cv2.contourArea(cnt)
                if area < 200:
                    continue
                
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * math.pi * (area / (perimeter * perimeter))
                if circularity < 0.65 or circularity > 1.0:
                    continue
                
                (cnt_cx, cnt_cy), cnt_r = cv2.minEnclosingCircle(cnt)
                center_distance = np.sqrt((cnt_cx - cx)**2 + (cnt_cy - cy)**2)
                radius_diff = abs(cnt_r - radius) / radius
                
                if center_distance > radius * 0.3 or radius_diff > 0.4:
                    continue
                
                score = circularity * (1 - radius_diff * 0.5) * (1 - center_distance/radius)
                if score > best_score:
                    best_score = score
                    best_outer = (cnt, area, cnt_cx, cnt_cy, cnt_r, circularity)
        
        if best_outer is None:
            return False, 0, 0, 0
        
        outer_cnt, outer_area, outer_cx, outer_cy, outer_r, outer_circularity = best_outer
        
        valid_holes = []
        for i, cnt in enumerate(contours):
            if hierarchy[0][i][3] != -1:
                hole_area = cv2.contourArea(cnt)
                if hole_area < 50:
                    continue
                
                (hx, hy), hr = cv2.minEnclosingCircle(cnt)
                distance_from_center = np.sqrt((hx - outer_cx)**2 + (hy - outer_cy)**2)
                if distance_from_center > outer_r * 0.4:
                    continue
                
                valid_holes.append((hr, hole_area, hx, hy))
        
        if not valid_holes:
            return False, 0, 0, 0
        
        best_hole = max(valid_holes, key=lambda x: x[1])
        inner_r, hole_area, hole_x, hole_y = best_hole
        
        ratio = (inner_r * 2) / (outer_r * 2)
        if ratio < 0.3 or ratio > 0.7:
            return False, 0, 0, 0
        
        final_score = (
            outer_circularity * 0.4 +
            min(1.0, hole_area / 500) * 0.3 +
            (1 - abs(ratio - 0.5) * 2) * 0.3
        )
        
        return True, outer_r * 2, inner_r * 2, final_score
    
    def _detect_gap_flexible(self, roi, center, outer_r, inner_r):
        x, y = int(center[0]), int(center[1])
        ring_radius = (outer_r + inner_r) / 2
        
        num_samples = 48
        angles = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
        
        intensities = []
        for angle in angles:
            sample_values = []
            for r_offset in [-2, 0, 2]:
                sample_r = ring_radius + r_offset
                px = int(x + sample_r * np.cos(angle))
                py = int(y + sample_r * np.sin(angle))
                
                if 0 <= px < roi.shape[1] and 0 <= py < roi.shape[0]:
                    sample_values.append(roi[py, px])
            
            if sample_values:
                intensities.append(np.mean(sample_values))
            else:
                intensities.append(0)
        
        if not intensities:
            return False, 0
        
        intensities = np.array(intensities)
        kernel = np.ones(3) / 3
        smoothed = np.convolve(intensities, kernel, mode='same')
        
        mean_val = np.mean(smoothed)
        std_val = np.std(smoothed)
        gap_threshold = mean_val + std_val * 0.5
        gap_mask = smoothed > gap_threshold
        
        gap_segments = []
        current_length = 0
        
        for i in range(len(gap_mask)):
            if gap_mask[i]:
                current_length += 1
            else:
                if current_length > 0:
                    gap_segments.append(current_length)
                    current_length = 0
        
        if current_length > 0:
            gap_segments.append(current_length)
        
        if not gap_segments:
            return False, 0
        
        max_gap_length = max(gap_segments)
        gap_ratio = max_gap_length / num_samples
        
        if (0.08 <= gap_ratio <= 0.4 and
            max_gap_length >= 3 and
            len(gap_segments) <= 3):
            return True, gap_ratio
        
        return False, 0
    
    def _detect_rings_raw(self, frame):
        processed = self._preprocess_for_ring(frame)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        processed_clean = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        
        circles = cv2.HoughCircles(
            processed_clean, cv2.HOUGH_GRADIENT,
            dp=self.dp, minDist=self.min_dist,
            param1=self.param1, param2=self.param2,
            minRadius=self.min_radius, maxRadius=self.max_radius
        )
        
        detected_rings = []
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            for x, y, r in circles[0, :]:
                margin = 15
                x1, y1 = max(0, x - r - margin), max(0, y - r - margin)
                x2, y2 = min(processed.shape[1], x + r + margin), min(processed.shape[0], y + r + margin)
                
                roi = processed[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                
                roi_center = (x - x1, y - y1)
                
                best_result = None
                best_confidence = 0
                
                threshold_methods = [
                    (cv2.THRESH_BINARY + cv2.THRESH_OTSU, False),
                    (cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU, False),
                ]
                
                for thresh_val, is_fixed in threshold_methods:
                    roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
                    _, binary = cv2.threshold(roi_blur, 0, 255, thresh_val)
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                    
                    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if len(contours) < 2:
                        continue
                    
                    is_valid_structure, outer_d, inner_d, structure_score = self._check_ring_structure(
                        contours, hierarchy, roi_center, r
                    )
                    
                    if not is_valid_structure or structure_score < 0.3:
                        continue
                    
                    has_gap, gap_ratio = self._detect_gap_flexible(
                        roi, roi_center, outer_d/2, inner_d/2
                    )
                    
                    if not has_gap:
                        continue
                    
                    total_confidence = structure_score * 0.7 + min(1.0, gap_ratio * 5) * 0.3
                    
                    if total_confidence > best_confidence and total_confidence > self.confidence_threshold:
                        best_confidence = total_confidence
                        best_result = (x, y, r, gap_ratio, total_confidence, outer_d, inner_d)
                
                if best_result:
                    detected_rings.append(best_result)
        
        return detected_rings
    
    def _update_landolt_tracking(self, current_detections):
        updated_tracked = []
        
        for tracked in self._landolt_tracking:
            matched = False
            best_match = None
            min_distance = float('inf')
            
            for detection in current_detections:
                dx, dy = detection[0] - tracked['x'], detection[1] - tracked['y']
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < self.tracking_distance_threshold and distance < min_distance:
                    min_distance = distance
                    best_match = detection
                    matched = True
            
            if matched and best_match:
                new_x = tracked['x'] * self.position_smoothing + best_match[0] * (1 - self.position_smoothing)
                new_y = tracked['y'] * self.position_smoothing + best_match[1] * (1 - self.position_smoothing)
                new_r = tracked['r'] * self.position_smoothing + best_match[2] * (1 - self.position_smoothing)
                
                updated_tracked.append({
                    'x': new_x,
                    'y': new_y,
                    'r': new_r,
                    'gap_ratio': best_match[3],
                    'confidence': best_match[4],
                    'outer_d': best_match[5],
                    'inner_d': best_match[6],
                    'frames_tracked': min(tracked['frames_tracked'] + 1, self.max_tracking_history),
                    'stable': tracked['frames_tracked'] >= self.min_tracking_frames,
                    'last_ocr_ids': tracked.get('last_ocr_ids', [])
                })
                
                current_detections.remove(best_match)
        
        for detection in current_detections:
            if detection[4] >= self.confidence_threshold:
                updated_tracked.append({
                    'x': detection[0],
                    'y': detection[1],
                    'r': detection[2],
                    'gap_ratio': detection[3],
                    'confidence': detection[4],
                    'outer_d': detection[5],
                    'inner_d': detection[6],
                    'frames_tracked': 1,
                    'stable': False,
                    'last_ocr_ids': []
                })
        
        self._landolt_tracking = [ring for ring in updated_tracked 
                                   if ring['frames_tracked'] >= self.min_tracking_frames or 
                                   ring['confidence'] > 0.6]
        
        if len(self._landolt_tracking) > 3:
            self._landolt_tracking.sort(key=lambda x: x['confidence'], reverse=True)
            self._landolt_tracking = self._landolt_tracking[:3]
    
    def _get_cache_key(self, x, y, r):
        return f"{int(x/10)*10}_{int(y/10)*10}_{int(r/5)*5}"
    
    def _ocr_cache_get(self, key, frame_count, timeout=10):
        if key in self._ocr_cache:
            cached_frame, cached_data = self._ocr_cache[key]
            if frame_count - cached_frame < timeout:
                self._ocr_cache_access[key] = frame_count
                return cached_data
        return None
    
    def _ocr_cache_put(self, key, frame_count, data):
        self._ocr_cache[key] = (frame_count, data)
        self._ocr_cache_access[key] = frame_count
        
        if len(self._ocr_cache) > self.ocr_cache_size:
            oldest_key = min(self._ocr_cache_access.keys(), 
                           key=lambda k: self._ocr_cache_access[k])
            del self._ocr_cache[oldest_key]
            del self._ocr_cache_access[oldest_key]
    
    def _ocr_around_ring_cached(self, frame, x, y, r, frame_count):
        cache_key = self._get_cache_key(x, y, r)
        cached = self._ocr_cache_get(cache_key, frame_count)
        
        if cached is not None:
            return cached
        
        expand_r = int(r * self.ocr_expand_factor)
        x1 = max(0, x - expand_r)
        y1 = max(0, y - expand_r)
        x2 = min(frame.shape[1], x + expand_r)
        y2 = min(frame.shape[0], y + expand_r)
        
        ocr_roi = frame[y1:y2, x1:x2]
        
        if ocr_roi.size == 0:
            return []
        
        detected_ids = []
        reader = self.models.easyocr_reader
        
        if reader is not None:
            try:
                results = reader.readtext(ocr_roi, detail=1, paragraph=False)
                
                seen = set()
                for box, text, conf in results:
                    alnum = re.sub(r'[^A-Za-z0-9]', '', text).upper()
                    if not alnum or conf < self.ocr_min_conf:
                        continue
                    if not ID_PATTERN.match(alnum):
                        continue
                    if alnum in seen:
                        continue
                    seen.add(alnum)
                    
                    detected_ids.append({
                        'id': alnum,
                        'confidence': conf
                    })
            except Exception as e:
                logger.warning(f"OCR error: {e}")
        
        self._ocr_cache_put(cache_key, frame_count, detected_ids)
        return detected_ids
    
    def _get_position_key(self, x, y):
        rounded_x = int(x / self.position_tolerance) * self.position_tolerance
        rounded_y = int(y / self.position_tolerance) * self.position_tolerance
        return f"{rounded_x}_{rounded_y}"
    
    def _should_auto_save(self, result):
        if not self.auto_save_enabled:
            return False
        
        if self.save_counter >= self.max_saves_per_session:
            return False
        
        ring_detected = True
        sufficient_confidence = result['ring_confidence'] >= self.min_ring_confidence
        is_stable = result['stable'] and result['frames_tracked'] >= self.min_stable_frames
        
        position_key = self._get_position_key(result['ring_x'], result['ring_y'])
        position_not_saved = position_key not in self.saved_positions
        
        should_save = (ring_detected and sufficient_confidence and 
                      is_stable and position_not_saved)
        
        return should_save
    
    def _auto_save_detection(self, frame, result, frame_count):
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if result['id_text']:
                id_part = f"_ID{result['id_text']}"
            else:
                id_part = "_NoID"
            
            filename_base = f"landolt_ring{id_part}_{timestamp}_frame{frame_count}"
            
            img_path = os.path.join(self.output_base, "landolt", "images", f"{filename_base}.jpg")
            data_path = os.path.join(self.output_base, "landolt", "data", f"{filename_base}_data.txt")
            
            output_frame = self._draw_landolt_results(frame.copy(), [result])
            cv2.imwrite(img_path, output_frame)
            
            with open(data_path, 'w') as f:
                f.write(f"=== LANDOLT RING AUTO SAVE DATA WITH ANALYTICS ===\n")
                f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Camera: {self.camera_id}\n")
                f.write(f"Frame: {frame_count}\n")
                f.write(f"Save Counter: {self.save_counter + 1}\n\n")
                
                f.write(f"--- RING DETECTION ---\n")
                f.write(f"Position: ({result['ring_x']}, {result['ring_y']})\n")
                f.write(f"Radius: {result['ring_r']}\n")
                f.write(f"Confidence: {result['ring_confidence']:.4f}\n")
                f.write(f"Gap Ratio: {result['ring_gap_ratio']:.4f}\n")
                f.write(f"Tracking Status: {'Stable' if result['stable'] else 'New'}\n")
                f.write(f"Frames Tracked: {result['frames_tracked']}\n\n")
                
                f.write(f"--- OCR DETECTION ---\n")
                if result['id_text']:
                    f.write(f"ID Text: {result['id_text']}\n")
                    f.write(f"ID Confidence: {result['id_confidence']:.4f}\n")
                else:
                    f.write(f"ID Text: Not detected\n")
                f.write("\n")
                
                f.write(f"--- CAMERA ANALYTICS (3D COORDINATES) ---\n")
                f.write(f"X (Distance): {result.get('x_distance', 0):.3f} m\n")
                f.write(f"Y (Lateral): {result.get('y_lateral', 0):.3f} m ({result.get('direction', 'unknown')})\n")
                f.write(f"Z (Height): {result.get('z_height', 0):.3f} m\n")
                f.write(f"Calibration Status: {'Calibrated' if self.camera_analytics.is_calibrated else 'Estimated'}\n")
            
            position_key = self._get_position_key(result['ring_x'], result['ring_y'])
            self.saved_positions.add(position_key)
            self.save_counter += 1
            
            print(f"\n[CAM {self.camera_id}] LANDOLT AUTO SAVE #{self.save_counter}")
            print(f"  Image: {img_path}")
            print(f"  Data: {data_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"[CAM {self.camera_id}] Auto save failed: {e}")
            return False
    
    def process_landolt(self, frame):
        self.frame_count += 1
        
        raw_detections = self._detect_rings_raw(frame)
        self._update_landolt_tracking(raw_detections)
        
        combined_results = []
        
        for ring in self._landolt_tracking:
            if ring['stable'] or ring['confidence'] > 0.5:
                x, y, r = int(ring['x']), int(ring['y']), int(ring['r'])
                
                should_run_ocr = (
                    not ring['stable'] or
                    ring['frames_tracked'] % 5 == 0 or
                    not ring['last_ocr_ids']
                )
                
                detected_ids = []
                if should_run_ocr:
                    detected_ids = self._ocr_around_ring_cached(frame, x, y, r, self.frame_count)
                    ring['last_ocr_ids'] = detected_ids
                else:
                    detected_ids = ring['last_ocr_ids']
                
                best_id = None
                if detected_ids:
                    best_id = max(detected_ids, key=lambda x: x['confidence'])
                
                x_distance, y_lateral, z_height, direction = self.camera_analytics.pixels_to_3d_coordinates(x, y, r)
                
                if not self.camera_analytics.is_calibrated and ring['stable']:
                    self.camera_analytics.calibrate_reference(r, 0.5)
                
                result = {
                    'ring_x': x,
                    'ring_y': y,
                    'ring_r': r,
                    'ring_confidence': ring['confidence'],
                    'ring_gap_ratio': ring['gap_ratio'],
                    'id_text': best_id['id'] if best_id else None,
                    'id_confidence': best_id['confidence'] if best_id else 0,
                    'frames_tracked': ring['frames_tracked'],
                    'stable': ring['stable'],
                    'x_distance': x_distance,
                    'y_lateral': y_lateral,
                    'z_height': z_height,
                    'direction': direction
                }
                
                combined_results.append(result)
                
                if self._should_auto_save(result):
                    self._auto_save_detection(frame, result, self.frame_count)
        
        self.current_landolt_results = combined_results
        # return combined_results
    
        if combined_results:  # Only store if there are results
            annotated = self.annotate_landolt(frame, combined_results)
            self.last_processed_frame = annotated
        self.last_processed_frame = annotated
        
        return combined_results
    
    def _draw_landolt_results(self, frame, results):
        for result in results:
            x, y, r = result['ring_x'], result['ring_y'], result['ring_r']
            has_id = result['id_text'] is not None
            
            if has_id:
                if result['stable']:
                    ring_color = (0, 255, 0)
                    id_box_color = (0, 255, 0)
                    thickness = 3
                else:
                    ring_color = (0, 255, 255)
                    id_box_color = (0, 255, 255)
                    thickness = 2
            else:
                ring_color = (0, 165, 255)
                id_box_color = (128, 128, 128)
                thickness = 2
            
            cv2.circle(frame, (x, y), r, ring_color, thickness, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1, cv2.LINE_AA)
            
            status = "[STABLE]" if result['stable'] else f"[{result['frames_tracked']}]"
            cv2.putText(frame, status, (x + r - 35, y - r + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, ring_color, 1, cv2.LINE_AA)
            
            ring_label = f"Ring: {result['ring_confidence']:.2f}"
            cv2.putText(frame, ring_label, (x - 30, y - r - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, ring_color, 1, cv2.LINE_AA)
            
            box_y = y + r + 20
            box_w, box_h = 120, 45
            
            cv2.rectangle(frame, (x - box_w//2, box_y), (x + box_w//2, box_y + box_h),
                         (0, 0, 0), -1)
            cv2.rectangle(frame, (x - box_w//2, box_y), (x + box_w//2, box_y + box_h),
                         id_box_color, 2)
            
            if has_id:
                cv2.putText(frame, f"ID: {result['id_text']}",
                           (x - box_w//2 + 8, box_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, id_box_color, 2, cv2.LINE_AA)
                cv2.putText(frame, f"Conf: {result['id_confidence']:.2f}",
                           (x - box_w//2 + 8, box_y + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, id_box_color, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "No ID", (x - box_w//2 + 25, box_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, id_box_color, 1, cv2.LINE_AA)
                cv2.putText(frame, "Processing...", (x - box_w//2 + 8, box_y + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, id_box_color, 1, cv2.LINE_AA)
        
        return frame

    def annotate_landolt(self, frame, results):
        annotated = self._draw_landolt_results(frame.copy(), results)
        
        if len(results) > 0:
            status_text = "STABLE - FROZEN"
            status_color = (0, 255, 0)
        elif len(self._landolt_tracking) > 0:
            status_text = "STABILIZING..."
            status_color = (0, 255, 255)
        else:
            status_text = "SCANNING..."
            status_color = (0, 0, 255)

        cv2.putText(annotated, status_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        h = annotated.shape[0]
        info_y = h - 60
        
        if results:
            info_text = f"Detected: {len(results)} ring(s)"
            if self.auto_save_enabled:
                info_text += f" | Auto-Save: ON ({self.save_counter}/{self.max_saves_per_session})"
            cv2.putText(annotated, info_text, (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    # ========================================================================
    # QR DETECTION
    # ========================================================================
    
    def process_qr(self, frame):
        detections = self.qr_detector_3d.process_qr_enhanced(frame)
        self.current_qr_detections = detections
        
        # ✅ TAMBAH INI
        if detections:
            annotated = self.annotate_qr(frame, detections)
            self.last_processed_frame = annotated
        
        return detections
    
    def annotate_qr(self, frame, detections):
        annotated = frame.copy()
        
        cv2.putText(annotated, "QR Detection", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        for detection in detections:
            qr_data = detection['qr_data']
            qr_points = detection['qr_points']
            xyz_info = detection['xyz_info']
            direction = self.qr_detector_3d._get_direction_text(xyz_info['angle_y'])
            
            if detection['direction_changed']:
                color = (0, 0, 255)
                status = f"DIR_CHANGE_{direction}"
            elif detection['detection_count'] >= self.qr_detector_3d.stabilization_frames:
                color = (0, 255, 0)
                status = f"STABLE_{direction}"
            else:
                color = (0, 255, 255)
                status = f"TRACKING_{direction} ({detection['detection_count']}/{self.qr_detector_3d.stabilization_frames})"
            
            pts = qr_points.astype(int).reshape(-1, 1, 2)
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=3)
            
            center = (int(xyz_info['center_x']), int(xyz_info['center_y']))
            cv2.circle(annotated, center, 5, color, -1)
            
            text_y = 50
            cv2.putText(annotated, f"Status: {status}", 
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            text_y += 25
            cv2.putText(annotated, f"X (Distance): {xyz_info['x_distance']:.2f}m", 
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            text_y += 25
            cv2.putText(annotated, f"Y ({direction}): {xyz_info['y_distance']:.2f}m", 
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            text_y += 25
            cv2.putText(annotated, f"Z (Height): {xyz_info['z_distance']:.2f}m", 
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            text_y += 25
            cv2.putText(annotated, f"Angle X: {xyz_info['angle_x']:.1f}° Y: {xyz_info['angle_y']:.1f}°", 
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if self.qr_detector_3d.qr_save_counter > 0:
                text_y += 25
                cv2.putText(annotated, f"Saves: {self.qr_detector_3d.qr_save_counter}", 
                           (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return annotated

    # ========================================================================
    # MOTION DETECTION - FULL IMPLEMENTATION
    # ========================================================================
    
    def process_motion(self, frame):
        """Process motion detection using full DotOnPlateTracker"""
        tracking_data = self.motion_tracker.process_frame(frame)
        self.current_motion_data = tracking_data
        
        if tracking_data:
                annotated = self.annotate_motion(frame, tracking_data)
                self.last_processed_frame = annotated
                
        return tracking_data
    
    def annotate_motion(self, frame, tracking_data):
        """Annotate frame with motion tracking data"""
        if tracking_data is None:
            annotated = frame.copy()
            cv2.putText(annotated, f"CAM {self.camera_id} - MOTION (No Detection)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return annotated
        
        annotated = self.motion_tracker.annotate_frame(frame, tracking_data)
        return annotated
        
    # ========================================================================
    # HAZMAT DETECTION
    # ========================================================================
    
    def _calculate_hazmat_real_world_coordinates(self, box):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        box_width = x2 - x1
        box_height = y2 - y1
        box_center_x = (x1 + x2) // 2
        box_center_y = (y1 + y2) // 2
        
        pixel_offset_x = box_center_x - self.frame_center_x
        horizontal_angle = (pixel_offset_x / self.frame_center_x) * (self.CAMERA_FOV_HORIZONTAL / 2)
        
        pixel_offset_y = self.frame_center_y - box_center_y
        vertical_angle = (pixel_offset_y / self.frame_center_y) * (self.CAMERA_FOV_VERTICAL / 2)
        
        object_area = box_width * box_height
        max_area = self.frame_width * self.frame_height * 0.3
        size_ratio = min(object_area / max_area, 1.0)
        distance_x = max(0.3, 2.0 - (size_ratio * 1.7))
        
        horizontal_y = distance_x * (horizontal_angle / 57.3)
        
        height_offset = distance_x * (vertical_angle / 57.3)
        height_z = self.HAZMAT_CAMERA_HEIGHT + height_offset
        height_z = max(0.0, min(3.0, height_z))
        
        if horizontal_angle < -0.5:
            direction = "Left"
        elif horizontal_angle > 0.5:
            direction = "Right"
        else:
            direction = "Center"
        
        return (round(distance_x, 2), round(horizontal_y, 2), round(height_z, 2),
                round(horizontal_angle, 1), round(vertical_angle, 1), direction)
    
    def _create_hazmat_object_id(self, box, class_name):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return f"{class_name}_{center_x//50}_{center_y//50}"
    
    def _save_hazmat_detection(self, frame, detections_data):
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            image_filename = f"{self.output_base}/hazmat/images/hazmat_cam{self.camera_id}_{timestamp}.jpg"
            cv2.imwrite(image_filename, frame)
            
            analytics_filename = f"{self.output_base}/hazmat/analytics/hazmat_cam{self.camera_id}_{timestamp}.txt"
            with open(analytics_filename, 'w') as f:
                f.write(f"Hazmat Detection Analysis\n")
                f.write(f"Camera: {self.camera_id}\n")
                f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Image File: {image_filename}\n")
                f.write(f"Total Objects Detected: {len(detections_data)}\n")
                f.write("=" * 60 + "\n\n")
                
                for i, detection in enumerate(detections_data, 1):
                    f.write(f"Object {i}:\n")
                    f.write(f"  Hazard Type: {detection['hazard_class']}\n")
                    f.write(f"  Confidence: {detection['confidence']}\n")
                    f.write(f"  Position Analysis:\n")
                    f.write(f"    Direction: {detection['direction']}\n")
                    f.write(f"    Distance X: {detection['distance_x']:.2f}m\n")
                    f.write(f"    Position Y: {detection['position_y']:+.2f}m\n")
                    f.write(f"    Height Z: {detection['height_z']:.2f}m\n")
                    f.write("-" * 40 + "\n\n")
            
            print(f"[CAM {self.camera_id}] Saved Hazmat: {image_filename}")
            return True
        except Exception as e:
            logger.error(f"[CAM {self.camera_id}] Hazmat save error: {e}")
            return False
    
    def process_hazmat(self, frame):
        try:
            if self.models.yolo_hazmat is None:
                return None
            
            with torch.no_grad():
                results = self.models.yolo_hazmat(frame, device=self.device, verbose=False)[0]
            
            detections_to_save = []
            current_objects = set()
            
            if results.boxes is not None:
                for box in results.boxes:
                    confidence = float(box.conf[0])
                    if confidence < self.HAZMAT_CONFIDENCE_THRESHOLD:
                        continue
                    
                    cls_id = int(box.cls[0])
                    class_name = results.names.get(cls_id, str(cls_id)) if hasattr(results, 'names') else str(cls_id)
                    hazard_label = self.hazard_classes.get(class_name, f"Unclassified: {class_name}")
                    
                    distance_x, position_y, height_z, horizontal_angle, vertical_angle, direction = \
                        self._calculate_hazmat_real_world_coordinates(box)
                    
                    object_id = self._create_hazmat_object_id(box, class_name)
                    current_objects.add(object_id)
                    
                    if object_id not in self.hazmat_detection_count:
                        self.hazmat_detection_count[object_id] = 0
                    self.hazmat_detection_count[object_id] += 1
                    
                    if (self.hazmat_detection_count[object_id] >= self.HAZMAT_STABILIZATION_FRAMES and
                        object_id not in self.hazmat_saved_objects):
                        
                        self.hazmat_saved_objects.add(object_id)
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detection_data = {
                            "object_id": object_id,
                            "hazard_class": hazard_label,
                            "confidence": round(confidence, 2),
                            "distance_x": distance_x,
                            "position_y": position_y,
                            "height_z": height_z,
                            "horizontal_angle": horizontal_angle,
                            "vertical_angle": vertical_angle,
                            "direction": direction,
                            "bounding_box": {
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2
                            }
                        }
                        detections_to_save.append(detection_data)
            
            objects_to_remove = set(self.hazmat_detection_count.keys()) - current_objects
            for obj_id in objects_to_remove:
                del self.hazmat_detection_count[obj_id]
                self.hazmat_saved_objects.discard(obj_id)
            
            if detections_to_save:
                annotated_frame = self._annotate_hazmat_for_save(frame.copy(), results)
                self._save_hazmat_detection(annotated_frame, detections_to_save)
            
            self.current_hazmat_results = results
            if results and results.boxes is not None:
                annotated = self._annotate_hazmat_for_save(frame.copy(), results)
                self.last_processed_frame = annotated
            return results
        except Exception as e:
            logger.error(f"[CAM {self.camera_id}] Hazmat error: {e}")
            return None
    
    def _annotate_hazmat_for_save(self, frame, results):
        annotated = frame.copy()
        
        if results.boxes is not None:
            for box in results.boxes:
                confidence = float(box.conf[0])
                if confidence < self.HAZMAT_CONFIDENCE_THRESHOLD:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                class_name = results.names.get(cls_id, str(cls_id)) if hasattr(results, 'names') else str(cls_id)
                hazard_label = self.hazard_classes.get(class_name, f"Unclassified: {class_name}")
                
                distance_x, position_y, height_z, _, _, _ = \
                    self._calculate_hazmat_real_world_coordinates(box)
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{hazard_label}",
                           (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(annotated, f"X:{distance_x:.2f}m Y:{position_y:+.2f}m Z:{height_z:.2f}m",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        return annotated
    
    # ========================================================================
    # CRACK DETECTION (abbreviated - keeping existing)
    # ========================================================================
    
    def process_crack(self, frame):
        try:
            if self.models.yolo_crack is None:
                return None
            
            square_points = self._detect_square(frame)
            if square_points is None:
                return None
            
            rect = self._order_points(square_points.reshape(4, 2))
            M = cv2.getPerspectiveTransform(
                np.float32(rect),
                np.float32([[0, 0], [self.WARP_SIZE, 0], 
                           [self.WARP_SIZE, self.WARP_SIZE], [0, self.WARP_SIZE]])
            )
            warped = cv2.warpPerspective(frame, M, (self.WARP_SIZE, self.WARP_SIZE))
            
            with torch.no_grad():
                results = self.models.yolo_crack.predict(warped, imgsz=640, verbose=False, device=self.device)
            
            crack_found = False
            self.crack_results_data.clear()
            overlayed = warped.copy()
            
            for r in results:
                if r.masks is not None:
                    masks = r.masks.data.cpu().numpy() if hasattr(r.masks.data, 'cpu') else r.masks.data
                    confidences = r.boxes.conf.cpu().numpy() if r.boxes is not None else [0.8] * len(masks)
                    
                    for idx, mask in enumerate(masks):
                        if np.sum(mask > 0.5) > 50:
                            crack_found = True
                            
                            mask_resized = cv2.resize((mask * 255).astype(np.uint8), 
                                                    (self.WARP_SIZE, self.WARP_SIZE))
                            
                            overlayed = self._apply_crack_mask(overlayed, mask_resized)
                            
                            crack_measurements = self._extract_crack_measurements(mask_resized)
                            for crack in crack_measurements:
                                if idx < len(confidences):
                                    crack.confidence = float(confidences[idx])
                                self.crack_results_data.append(crack)
            
            if crack_found:
                cv2.rectangle(overlayed, (0, 0), (self.WARP_SIZE-1, self.WARP_SIZE-1), (0, 255, 0), 3)
                
                clean_warped = warped.copy()
                cv2.rectangle(clean_warped, (0, 0), (self.WARP_SIZE-1, self.WARP_SIZE-1), (0, 0, 0), 1)

                
                display = self._create_crack_display(overlayed, clean_warped)
                
                self.current_crack_analysis = {
                    'warped': warped,
                    'overlayed': overlayed,
                    'display': display,
                    'square': square_points,
                    'original_frame': frame.copy()
                }
                return self.current_crack_analysis
            
            return None
        except Exception as e:
            logger.error(f"[CAM {self.camera_id}] Crack error: {e}")
            return None
    
    def _apply_crack_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mask_bool = mask > 127
        if np.any(mask_bool):
            frame[mask_bool] = [0, 0, 255]
        return frame
    
    def _extract_crack_measurements(self, mask: np.ndarray) -> List[CrackData]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < 50:
            return []
        
        cnt = largest_contour
        x, y, w, h = cv2.boundingRect(cnt)
        
        width_m = w * self.M_PER_PIXEL
        height_m = h * self.M_PER_PIXEL
        length_m = np.sqrt(width_m**2 + height_m**2)
        
        crack_region = mask[y:y+h, x:x+w]
        if crack_region.size > 0:
            dist_transform = cv2.distanceTransform(crack_region, cv2.DIST_L2, 3)
            max_thickness = np.max(dist_transform) * 2 * self.M_PER_PIXEL
            non_zero_distances = dist_transform[dist_transform > 0]
            min_thickness = np.min(non_zero_distances) * 2 * self.M_PER_PIXEL if len(non_zero_distances) > 0 else 0.0
        else:
            max_thickness = 0.0
            min_thickness = 0.0
        
        endpoints = self._find_crack_endpoints(cnt)
        
        endpoints_world = []
        for ep in endpoints:
            pixel_x_from_center = ep[0] - (self.WARP_SIZE / 2)
            pixel_y_from_center = ep[1] - (self.WARP_SIZE / 2)
            world_x = pixel_x_from_center * self.M_PER_PIXEL
            world_y = pixel_y_from_center * self.M_PER_PIXEL
            endpoints_world.append((world_x, world_y))
        
        crack_data = CrackData(
            x=x, y=y, w=w, h=h,
            width=width_m, height=height_m, length=length_m,
            max_thickness=max_thickness, min_thickness=min_thickness,
            endpoints=endpoints, endpoints_world_coords=endpoints_world
        )
        
        return [crack_data]
    
    def _find_crack_endpoints(self, contour: np.ndarray) -> List[Tuple[int, int]]:
        cnt = contour.reshape(-1, 2)
        
        leftmost = tuple(cnt[cnt[:, 0].argmin()])
        rightmost = tuple(cnt[cnt[:, 0].argmax()])
        topmost = tuple(cnt[cnt[:, 1].argmin()])
        bottommost = tuple(cnt[cnt[:, 1].argmax()])
        
        points = [leftmost, rightmost, topmost, bottommost]
        max_dist = 0
        best_pair = points[:2]
        
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
                if dist > max_dist:
                    max_dist = dist
                    best_pair = [points[i], points[j]]
        
        return best_pair
    
    def _calculate_spatial_position(self, crack: CrackData) -> dict:
        center_x = crack.x + crack.w // 2
        center_y = crack.y + crack.h // 2
        
        pixel_x_from_center = center_x - (self.WARP_SIZE / 2)
        pixel_y_from_center = center_y - (self.WARP_SIZE / 2)
        
        horizontal_offset_meters = pixel_x_from_center * self.M_PER_PIXEL
        
        max_horizontal_pixels = self.WARP_SIZE / 2
        pixel_angle_ratio = self.CAMERA_FOV_H / (2 * max_horizontal_pixels)
        y_angle_from_camera = pixel_x_from_center * pixel_angle_ratio
        
        vertical_pixel_angle_ratio = self.CAMERA_FOV_V / (2 * max_horizontal_pixels)
        vertical_angle = pixel_y_from_center * vertical_pixel_angle_ratio
        
        distance_from_camera = self.DETECTED_SQUARE_SIZE / (2 * math.tan(math.radians(self.CAMERA_FOV_H / 2)))
        distance_offset = pixel_y_from_center * self.M_PER_PIXEL * 0.1
        distance_from_camera += distance_offset
        
        height_from_ground = self.CAMERA_HEIGHT + (pixel_y_from_center * self.M_PER_PIXEL * 0.2)
        
        direction = "Center"
        if y_angle_from_camera < -2:
            direction = "Left"
        elif y_angle_from_camera > 2:
            direction = "Right"
        
        return {
            'direction': direction,
            'distance_from_camera': distance_from_camera,
            'horizontal_distance': horizontal_offset_meters,
            'height_from_ground': height_from_ground,
            'horizontal_angle': y_angle_from_camera,
            'vertical_angle': vertical_angle,
            'center_point': (center_x, center_y),
            'camera_height': self.CAMERA_HEIGHT,
            'detected_square_size': self.DETECTED_SQUARE_SIZE
        }
    
    def _create_crack_display(self, crack_img: np.ndarray, clean_img: np.ndarray) -> np.ndarray:
        h, w = crack_img.shape[:2]
        margin = 50
        gap = 50
        total_width = margin + w + gap + w + margin
        total_height = h + 2 * margin
        combined = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
        
        combined[margin:margin + h, margin:margin + w] = crack_img
        right_start = margin + w + gap
        combined[margin:margin + h, right_start:right_start + w] = clean_img
        
        self._draw_crack_annotations(combined, margin, right_start, w, h)
        return combined
    
    def _draw_crack_annotations(self, combined_img: np.ndarray, left_margin: int, right_start: int, img_w: int, img_h: int):
        margin = 50
        
        for crack in self.crack_results_data:
            if len(crack.endpoints) >= 2:
                ep1, ep2 = crack.endpoints
                
                ep1_adj = (ep1[0] + right_start, ep1[1] + margin)
                ep2_adj = (ep2[0] + right_start, ep2[1] + margin)
                
                frame_right = combined_img.shape[1]
                frame_bottom = combined_img.shape[0]
                
                for i, ep_adj in enumerate([ep1_adj, ep2_adj]):
                    for x in range(ep_adj[0], frame_right - 25, 4):
                        cv2.line(combined_img, (x, ep_adj[1]), (x + 2, ep_adj[1]), (128, 128, 128), 1)
                    
                    for y in range(ep_adj[1], frame_bottom - 25, 4):
                        cv2.line(combined_img, (ep_adj[0], y), (ep_adj[0], y + 2), (128, 128, 128), 1)
                    
                    cv2.circle(combined_img, ep_adj, 4, (255, 0, 0), -1)
                    cv2.circle(combined_img, ep_adj, 6, (0, 0, 0), 1)
                    cv2.putText(combined_img, f"EP{i+1}", (ep_adj[0] + 8, ep_adj[1] - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                cv2.arrowedLine(combined_img, ep1_adj, ep2_adj, (0, 0, 0), 2, tipLength=0.05)
                cv2.arrowedLine(combined_img, ep2_adj, ep1_adj, (0, 0, 0), 2, tipLength=0.05)
                
                spatial_info = self._calculate_spatial_position(crack)
                info_text = f"X:{spatial_info['distance_from_camera']:.3f}m Y:{spatial_info['horizontal_distance']:+.3f}m Z:{spatial_info['height_from_ground']:.3f}m | {crack.classification} | Conf:{crack.confidence:.2f}"
                if crack.is_critical:
                    info_text += " | CRITICAL"
                
                cv2.putText(combined_img, info_text, (left_margin, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0) if crack.is_critical else (0, 0, 0), 1)
                
                break
    
    def _save_crack_report(self, timestamp: str, output_folder: str):
        txt_path = os.path.join(output_folder, f"crack_report_{timestamp}.txt")
        
        with open(txt_path, 'w') as f:
            f.write("Crack Detection Analysis\n")
            f.write(f"Camera: {self.camera_id}\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Cracks Detected: {len(self.crack_results_data)}\n")
            f.write("=" * 60 + "\n\n")
            
            for idx, crack in enumerate(self.crack_results_data, 1):
                spatial_info = self._calculate_spatial_position(crack)
                
                f.write(f"Crack {idx}:\n")
                f.write(f"Classification: {crack.classification}\n")
                f.write(f"Confidence: {crack.confidence:.2f}\n\n")
                
                f.write("POSITION ANALYSIS (METERS):\n")
                f.write(f"X - Distance from Camera: {spatial_info['distance_from_camera']:.3f} m\n")
                f.write(f"Y - Horizontal Distance: {spatial_info['horizontal_distance']:+.3f} m\n")
                f.write(f"Z - Height from Ground: {spatial_info['height_from_ground']:.3f} m\n\n")
                
                f.write("CRACK MEASUREMENTS:\n")
                f.write(f"Width: {crack.width:.4f} m\n")
                f.write(f"Height: {crack.height:.4f} m\n")
                f.write(f"Length: {crack.length:.4f} m\n")
                f.write(f"Max Thickness: {crack.max_thickness:.4f} m\n")
                f.write(f"Critical: {'Yes' if crack.is_critical else 'No'}\n")
                f.write("-" * 40 + "\n")
        
        print(f"[CAM {self.camera_id}] Crack report saved: {txt_path}")
    
    def _detect_square(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cv2.contourArea(cnt) > 5000:
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                    if len(approx) == 4:
                        return approx
        except Exception as e:
            logger.error(f"[CAM {self.camera_id}] Square detection error: {e}")
        return None
    
    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()
        rect[0] = pts[np.argmin(s)]
        rect[1] = pts[np.argmin(d)]
        rect[2] = pts[np.argmax(s)]
        rect[3] = pts[np.argmax(d)]
        return rect
    
    # ========================================================================
    # RUST DETECTION - FULL IMPLEMENTATION FROM RUSTINI.PY
    # ========================================================================
    
    def process_rust(self, frame):
        """Enhanced rust detection with auto-save and full analytics"""
        try:
            if self.models.deeplabv3_rust is None:
                return None
            
            square_points = self._detect_square(frame)
            if square_points is None:
                self.rust_detection_count.clear()
                self.rust_saved_objects.clear()
                return None
            
            # Calculate position in METERS from ground
            self.rust_position_data = self._calculate_rust_position(square_points, frame.shape)
            
            # Warp perspective
            rect = self._order_points(square_points.reshape(4, 2))
            M = cv2.getPerspectiveTransform(
                np.float32(rect),
                np.float32([[0, 0], [self.RUST_WARP_SIZE, 0], 
                           [self.RUST_WARP_SIZE, self.RUST_WARP_SIZE], [0, self.RUST_WARP_SIZE]])
            )
            warped = cv2.warpPerspective(frame, M, (self.RUST_WARP_SIZE, self.RUST_WARP_SIZE))
            
            # Process with DeepLabV3 model
            image_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
            input_tensor = transform(image_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.models.deeplabv3_rust(input_tensor)['out']
                mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy().astype(np.uint8)
            
            mask_resized = cv2.resize(mask, (self.RUST_WARP_SIZE, self.RUST_WARP_SIZE), 
                                     interpolation=cv2.INTER_NEAREST)
            
            # Extract corrosion measurements
            self.rust_results_data = self._extract_rust_measurements(mask_resized)
            
            # Create visualizations
            overlayed = warped.copy()
            for class_id, color in self.rust_class_colors.items():
                overlayed[mask_resized == class_id] = color
            
            cv2.rectangle(overlayed, (0, 0), (self.RUST_WARP_SIZE-1, self.RUST_WARP_SIZE-1), 
                         (0, 255, 0), 3)
            
            clean = warped.copy()
            cv2.rectangle(clean, (0, 0), (self.RUST_WARP_SIZE-1, self.RUST_WARP_SIZE-1), 
                         (0, 0, 0), 1)
            
            display = self._create_rust_display(overlayed, clean)
            
            # Track detection and auto-save
            if self.rust_results_data:
                object_id = self._create_rust_object_id(self.rust_results_data[0])
                
                if object_id not in self.rust_detection_count:
                    self.rust_detection_count[object_id] = 0
                self.rust_detection_count[object_id] += 1
                
                # Auto-save when stable
                if (self.rust_detection_count[object_id] >= self.RUST_STABILIZATION_FRAMES and
                    object_id not in self.rust_saved_objects):
                    self._save_rust_detection(frame, warped, display, square_points)
                    self.rust_saved_objects.add(object_id)
            
            self.current_rust_analysis = {
                'warped': warped,
                'overlayed': overlayed,
                'display': display,
                'mask': mask_resized,
                'square': square_points,
                'original_frame': frame.copy()
            }
            
            self.last_processed_frame = display
            return self.current_rust_analysis
            
        except Exception as e:
            logger.error(f"[CAM {self.camera_id}] Rust error: {e}")
            return None
    
    def _calculate_rust_position(self, square_points: np.ndarray, frame_shape: tuple) -> RustPositionData:
        """Calculate position in METERS from ground level (0 = ground)"""
        points = square_points.reshape(4, 2)
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        
        # Calculate square size in pixels
        side1_length = np.linalg.norm(points[1] - points[0])
        side2_length = np.linalg.norm(points[2] - points[1])
        square_size_pixels = (side1_length + side2_length) / 2
        
        # X: Distance from camera in METERS
        x_distance_m = (self.RUST_REAL_SQUARE_SIZE * self.RUST_CAMERA_FOCAL_LENGTH) / square_size_pixels
        
        # Calculate offsets
        camera_center_x = self.frame_width / 2
        camera_center_y = self.frame_height / 2
        x_offset_pixels = center_x - camera_center_x
        y_offset_pixels = center_y - camera_center_y
        
        # Meters per pixel at this distance
        m_per_pixel = x_distance_m / self.RUST_CAMERA_FOCAL_LENGTH
        
        # Y: Horizontal position in METERS (+ = right, - = left)
        y_distance_m = x_offset_pixels * m_per_pixel
        
        # Z: Height from GROUND in METERS (0 = ground level)
        z_offset_from_camera = -y_offset_pixels * m_per_pixel
        z_height_ground_m = max(0.0, self.RUST_CAMERA_HEIGHT + z_offset_from_camera)
        
        # Angles in degrees
        angle_horizontal = math.degrees(math.atan2(y_distance_m, x_distance_m))
        angle_vertical = math.degrees(math.atan2(z_offset_from_camera, x_distance_m))
        
        estimated_square_size = (square_size_pixels * x_distance_m) / self.RUST_CAMERA_FOCAL_LENGTH
        
        return RustPositionData(
            x_distance_m=x_distance_m,
            y_distance_m=y_distance_m,
            z_height_ground_m=z_height_ground_m,
            square_size_m=estimated_square_size,
            angle_horizontal=angle_horizontal,
            angle_vertical=angle_vertical
        )
    
    def _extract_rust_measurements(self, mask: np.ndarray) -> List[CorrosionData]:
        """Extract corrosion severity measurements from mask"""
        total_pixels = mask.shape[0] * mask.shape[1]
        severity_percentages = {}
        total_affected = 0
        
        for class_id in range(1, self.RUST_NUM_CLASSES):
            count = np.sum(mask == class_id)
            percentage = (count / total_pixels) * 100
            if percentage > 0.1:
                severity_percentages[class_id] = percentage
                total_affected += count
        
        if not severity_percentages:
            return []
        
        total_affected_percentage = (total_affected / total_pixels) * 100
        return [CorrosionData(severity_percentages, total_affected_percentage)]
    
    def _create_rust_display(self, corrosion_img: np.ndarray, clean_img: np.ndarray) -> np.ndarray:
        """Create side-by-side display with corrosion analysis"""
        h, w = corrosion_img.shape[:2]
        margin, gap = 50, 50
        combined = np.ones((h + 2 * margin, margin + w + gap + w + margin, 3), dtype=np.uint8) * 255
        
        # Place images
        combined[margin:margin + h, margin:margin + w] = corrosion_img
        right_start = margin + w + gap
        combined[margin:margin + h, right_start:right_start + w] = clean_img
        
        if self.rust_results_data:
            corrosion = self.rust_results_data[0]
            
            # Header info
            info_text = f"Total Affected: {corrosion.total_affected_percentage:.1f}% | Dominant: {corrosion.dominant_class}"
            if corrosion.is_critical:
                info_text += " | CRITICAL"
            cv2.putText(combined, info_text, (margin, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 0, 0) if corrosion.is_critical else (0, 0, 0), 1)
            
            # Severity breakdown with semi-transparent background
            class_names = {1: "Fair", 2: "Poor", 3: "Severe"}
            severity_lines = []
            for class_id, percentage in corrosion.severity_percentages.items():
                severity_lines.append((f"{class_names[class_id]}: {percentage:.1f}%", 
                                      self.rust_class_colors[class_id]))
            
            if severity_lines:
                box_x = margin + w - 120
                box_y = margin + 5
                
                # Semi-transparent white background
                overlay = combined.copy()
                cv2.rectangle(overlay, (box_x - 5, box_y), 
                             (box_x + 115, box_y + len(severity_lines) * 20 + 5),
                             (255, 255, 255), -1)
                combined = cv2.addWeighted(overlay, 0.8, combined, 0.2, 0)
                
                # Add severity text
                text_y = box_y + 15
                for text, color in severity_lines:
                    cv2.putText(combined, text, (box_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    text_y += 20
            
            # Image labels
            cv2.putText(combined, "Detected Corrosion", (margin, margin + h + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(combined, "Original Square", (right_start, margin + h + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return combined
    
    def _create_rust_object_id(self, corrosion_data: CorrosionData) -> str:
        """Create unique ID for rust detection based on severity"""
        return f"rust_{corrosion_data.dominant_class}_{int(corrosion_data.total_affected_percentage)}"
    
    def _save_rust_detection(self, original_frame, warped, display, square_points):
        """Save rust detection with full analytics including position data"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save display image
            img_filename = f"{self.output_base}/rust/images/rust_cam{self.camera_id}_{timestamp}.jpg"
            cv2.imwrite(img_filename, display)
            
            # Save analytics text file
            analytics_filename = f"{self.output_base}/rust/analytics/rust_cam{self.camera_id}_{timestamp}.txt"
            
            with open(analytics_filename, 'w', encoding='utf-8') as f:
                f.write(f"=== RUST/CORROSION DETECTION ANALYSIS ===\n")
                f.write(f"Camera: {self.camera_id}\n")
                f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n\n")
                
                if self.rust_results_data:
                    corrosion = self.rust_results_data[0]
                    
                    f.write(f"CORROSION ANALYSIS:\n")
                    f.write(f"Total Affected: {corrosion.total_affected_percentage:.1f}%\n")
                    f.write(f"Dominant Class: {corrosion.dominant_class}\n")
                    f.write(f"Critical Status: {'CRITICAL' if corrosion.is_critical else 'Normal'}\n\n")
                    
                    f.write(f"Severity Breakdown:\n")
                    class_names = {1: "Fair", 2: "Poor", 3: "Severe"}
                    for class_id, percentage in corrosion.severity_percentages.items():
                        f.write(f"  {class_names[class_id]}: {percentage:.1f}%\n")
                    f.write("\n")
                
                if self.rust_position_data:
                    f.write(f"POSITION ANALYSIS (METERS from ground level 0,0):\n")
                    f.write(f"X - Distance from Camera: {self.rust_position_data.x_distance_m:.3f} m\n")
                    f.write(f"Y - Horizontal Distance: {self.rust_position_data.y_distance_m:.3f} m ")
                    f.write(f"({'Right' if self.rust_position_data.y_distance_m > 0 else 'Left'})\n")
                    f.write(f"Z - Height from Ground: {self.rust_position_data.z_height_ground_m:.3f} m\n")
                    f.write(f"Camera Height: {self.RUST_CAMERA_HEIGHT:.3f} m (above ground)\n")
                    f.write(f"Detected Square Size: {self.rust_position_data.square_size_m:.3f} m\n")
                    f.write(f"Horizontal Angle: {self.rust_position_data.angle_horizontal:.1f}°\n")
                    f.write(f"Vertical Angle: {self.rust_position_data.angle_vertical:.1f}°\n\n")
                    
                    # Position interpretation
                    f.write(f"POSITION INTERPRETATION:\n")
                    if self.rust_position_data.x_distance_m < 0.20:
                        f.write(f"  - Object is VERY CLOSE ({self.rust_position_data.x_distance_m:.3f}m)\n")
                    elif self.rust_position_data.x_distance_m < 0.50:
                        f.write(f"  - Object is at CLOSE range ({self.rust_position_data.x_distance_m:.3f}m)\n")
                    elif self.rust_position_data.x_distance_m < 1.00:
                        f.write(f"  - Object is at MEDIUM range ({self.rust_position_data.x_distance_m:.3f}m)\n")
                    else:
                        f.write(f"  - Object is at FAR range ({self.rust_position_data.x_distance_m:.3f}m)\n")
                    
                    if abs(self.rust_position_data.y_distance_m) < 0.02:
                        f.write(f"  - Object is CENTERED horizontally\n")
                    elif abs(self.rust_position_data.y_distance_m) < 0.10:
                        direction = "RIGHT" if self.rust_position_data.y_distance_m > 0 else "LEFT"
                        f.write(f"  - Object is slightly to the {direction} ({abs(self.rust_position_data.y_distance_m):.3f}m)\n")
                    else:
                        direction = "RIGHT" if self.rust_position_data.y_distance_m > 0 else "LEFT"
                        f.write(f"  - Object is significantly to the {direction} ({abs(self.rust_position_data.y_distance_m):.3f}m)\n")
                    f.write("\n")
                
                f.write(f"FILES SAVED:\n")
                f.write(f"  Image: {os.path.basename(img_filename)}\n")
                f.write(f"  Analytics: {os.path.basename(analytics_filename)}\n")
            
            print(f"\n[CAM {self.camera_id}] RUST SAVED")
            print(f"  Image: {img_filename}")
            print(f"  Analytics: {analytics_filename}")
            if self.rust_position_data:
                print(f"  Position: {self.rust_position_data}")
            if self.rust_results_data:
                corrosion = self.rust_results_data[0]
                print(f"  Severity: {corrosion.total_affected_percentage:.1f}% ({corrosion.dominant_class})")
                if corrosion.is_critical:
                    print(f"  Status: CRITICAL!")
            
            return True
            
        except Exception as e:
            logger.error(f"[CAM {self.camera_id}] Rust save error: {e}")
            return False
    
    # ========================================================================
    # SAVE FUNCTIONALITY
    # ========================================================================
    def update_position_display(self, position):
        """Update position-related UI elements."""
        if not position:
            return
            
        try:
            # Update position labels
            self.ui.DronePositionX.setText(f"[{position['x']:.2f}] m")
            self.ui.DronePositionY.setText(f"[{position['y']:.2f}] m") 
            self.ui.DroneHeight.setText(f"{position['z']:.2f} meter")
            
            # Update altitude slider (scale for 0-300 range)
            altitude_value = int(max(0, min(300, abs(position['z']) * 10)))
            self.ui.DroneAltitude.setValue(altitude_value)
            
        except Exception as e:
            self.log_debug(f"Error updating position: {e}")
            
    def _get_latest_frame_for_mode(self, mode):
        """
        Get latest frame based on current mode
        Priority: last_processed_frame > raw_frame > mode-specific results
        """
        # Priority 1: Last processed/annotated frame
        if hasattr(self, 'last_processed_frame') and self.last_processed_frame is not None:
            print(f"[CAM {self.camera_id}] ✓ Using last processed frame")
            return self.last_processed_frame
        
        # Priority 2: Raw camera frame (PENTING!)
        if hasattr(self, 'raw_frame') and self.raw_frame is not None:
            print(f"[CAM {self.camera_id}] ✓ Using raw camera frame")
            return self.raw_frame
        
        # Priority 3: Mode-specific results (fallback)
        if mode == 'crack' and self.current_crack_analysis:
            print(f"[CAM {self.camera_id}] ✓ Using crack analysis display")
            return self.current_crack_analysis.get('display')
        
        elif mode == 'rust' and self.current_rust_analysis:
            print(f"[CAM {self.camera_id}] ✓ Using rust analysis display")
            return self.current_rust_analysis.get('display')
        
        # No frame available
        print(f"[CAM {self.camera_id}] ❌ No frame available (no raw_frame, no last_processed_frame)")
        return None
    
    
    def _create_annotated_frame_for_save(self, mode):
        """
        DEPRECATED - Not used anymore since we store frames in process_frames
        """
        return self._get_latest_frame_for_mode(mode)
    
    
    def save_current_detection(self, mode: str, force_save=False):
        """
        ALWAYS save - priority: frozen > last_processed_frame > fallback
        
        Args:
            mode: Detection mode ('qr', 'landolt', 'hazmat', etc.)
            force_save: Always True (kept for compatibility)
        
        Returns:
            bool: Success status
        """
        # Priority 1: Frozen frame (detection stable)
        if self.is_frozen and self.frozen_frame is not None:
            frame_to_save = self.frozen_frame
            save_type = "FROZEN"
            print(f"[CAM {self.camera_id}] 📸 Saving FROZEN frame")
        
        # Priority 2: Last processed frame (live/no detection)
        elif hasattr(self, 'last_processed_frame') and self.last_processed_frame is not None:
            frame_to_save = self.last_processed_frame
            save_type = "LIVE"
            print(f"[CAM {self.camera_id}] 📸 Saving LIVE frame (snapshot)")
        
        # Priority 3: Try mode-specific frame
        else:
            frame_to_save = self._get_latest_frame_for_mode(mode)
            if frame_to_save is None:
                print(f"[CAM {self.camera_id}] ❌ No frame available in memory")
                return False
            save_type = "FALLBACK"
            print(f"[CAM {self.camera_id}] 📸 Saving FALLBACK frame")
        
        # ===== START SAVE PROCESS =====
        import csv
        import datetime
        import os
        import math
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Color mapping for CSV
        color_map = {
            'landolt': 'yellow',
            'qr': 'blue',
            'hazmat': 'red',
            'crack': 'orange',
            'rust': 'brown',
            'motion': 'green'
        }
        
        # Camera offset distances
        CAMERA_OFFSETS = {0: 0.10, 1: 0.10, 2: 0.10}
        offset_distance = CAMERA_OFFSETS.get(self.camera_id, 0.10)
        
        def calculate_offset_position(drone_x, drone_y, drone_z, yaw_rad, offset_m):
            """Calculate detection position offset in front of drone."""
            if drone_x is None or drone_y is None or yaw_rad is None:
                return drone_x, drone_y, drone_z
            
            offset_x = drone_x + (offset_m * math.cos(yaw_rad))
            offset_y = drone_y + (offset_m * math.sin(yaw_rad))
            offset_z = drone_z if drone_z is not None else None
            return offset_x, offset_y, offset_z
        
        # Get drone telemetry data
        drone_x, drone_y, drone_z, drone_yaw = None, None, None, None
        try:
            record = self.drone_parser.get_latest()
            self.drone_parser.save_data()
            self.drone_parser.stop()
            
            if record:
                pos = self.drone_parser.get_position()
                rpy = self.drone_parser.get_rpy()
                
                if pos:
                    drone_x, drone_y, drone_z = pos['x'], pos['y'], pos['z']
                    print(f"[CAM {self.camera_id}] ✓ Position: X={drone_x:.2f}, Y={drone_y:.2f}, Z={drone_z:.2f}")
                
                if rpy:
                    drone_yaw = rpy['yaw']
                    print(f"[CAM {self.camera_id}] ✓ Yaw: {drone_yaw:.3f} rad ({rpy['yaw_deg']:.1f}°)")
        
        except Exception as e:
            print(f"[CAM {self.camera_id}] ⚠️ Telemetry error: {e}")
        
        # Calculate detection position
        det_x, det_y, det_z = calculate_offset_position(
            drone_x, drone_y, drone_z, drone_yaw, offset_distance
        )
        position_str = f"({det_x:.2f}, {det_y:.2f}, {det_z:.2f})" if det_x is not None else "(None, None, None)"
        
        # ===== MODE-SPECIFIC SAVE LOGIC =====
        data_field = "N/A"
        base_path = self.output_base
        
        if mode == 'qr':
            filename = f"{base_path}/qr/images/qr_cam{self.camera_id}_{timestamp}.jpg"
            cv2.imwrite(filename, frame_to_save)
            self.qr_counter += 1
            
            if self.current_qr_detections and len(self.current_qr_detections) > 0:
                qr_data = self.current_qr_detections[0]['qr_data']
                data_field = f"QRCode {self.qr_counter}: {qr_data}"
            else:
                data_field = f"QRCode {self.qr_counter}: No detection (snapshot)"
            
            print(f"[CAM {self.camera_id}] ✅ Saved QR ({save_type}): {filename}")
        
        elif mode == 'landolt':
            filename = f"{base_path}/landolt/images/landolt_cam{self.camera_id}_{timestamp}.jpg"
            cv2.imwrite(filename, frame_to_save)
            self.landolt_counter += 1
            
            if self.current_landolt_results and len(self.current_landolt_results) > 0:
                best = max(self.current_landolt_results, key=lambda x: x['ring_confidence'])
                id_text = best.get('id_text', None)
                id_conf = best.get('id_confidence', 0)
                
                if id_text:
                    data_field = f"Landolt {self.landolt_counter}: ID {id_text} (Conf: {id_conf:.2f})"
                else:
                    data_field = f"Landolt {self.landolt_counter}: No ID"
            else:
                data_field = f"Landolt {self.landolt_counter}: No detection (snapshot)"
            
            print(f"[CAM {self.camera_id}] ✅ Saved Landolt ({save_type}): {filename}")
        
        elif mode == 'hazmat':
            filename = f"{base_path}/hazmat/images/hazmat_cam{self.camera_id}_{timestamp}.jpg"
            cv2.imwrite(filename, frame_to_save)
            self.hazmat_counter += 1
            
            if self.current_hazmat_results and self.current_hazmat_results.boxes is not None:
                boxes = [box for box in self.current_hazmat_results.boxes 
                         if float(box.conf[0]) >= self.HAZMAT_CONFIDENCE_THRESHOLD]
                if boxes:
                    box = boxes[0]
                    cls_id = int(box.cls[0])
                    class_name = self.current_hazmat_results.names.get(cls_id, str(cls_id))
                    hazard = self.hazard_classes.get(class_name, class_name)
                    data_field = f"Hazmat {self.hazmat_counter}: {hazard}"
                else:
                    data_field = f"Hazmat {self.hazmat_counter}: No detection (snapshot)"
            else:
                data_field = f"Hazmat {self.hazmat_counter}: No detection (snapshot)"
            
            print(f"[CAM {self.camera_id}] ✅ Saved Hazmat ({save_type}): {filename}")
        
        elif mode == 'crack' and self.current_crack_analysis:
            capture_folder = f"{base_path}/crack/crack_capture_{timestamp}"
            os.makedirs(capture_folder, exist_ok=True)
            
            cv2.imwrite(f"{capture_folder}/crack_original_{timestamp}.jpg", 
                       self.current_crack_analysis['original_frame'])
            cv2.imwrite(f"{capture_folder}/crack_analysis_{timestamp}.jpg", 
                       self.current_crack_analysis['display'])
            
            self.crack_counter += 1
            
            if self.crack_results_data and len(self.crack_results_data) > 0:
                crack = self.crack_results_data[0]
                data_field = f"Crack {self.crack_counter}: {crack.length:.4f}m ({crack.classification})"
            else:
                data_field = f"Crack {self.crack_counter}: Unknown (snapshot)"
            
            print(f"[CAM {self.camera_id}] ✅ Saved Crack ({save_type}): {capture_folder}")
        
        elif mode == 'rust' and self.current_rust_analysis:
            filename = f"{base_path}/rust/images/rust_cam{self.camera_id}_{timestamp}.jpg"
            cv2.imwrite(filename, frame_to_save)
            
            analysis_file = f"{base_path}/rust/images/rust_analysis_cam{self.camera_id}_{timestamp}.jpg"
            cv2.imwrite(analysis_file, self.current_rust_analysis['display'])
            
            self.rust_counter += 1
            
            if self.rust_results_data and len(self.rust_results_data) > 0:
                corrosion = self.rust_results_data[0]
                data_field = f"Rust {self.rust_counter}: {corrosion.dominant_class} ({corrosion.total_affected_percentage:.1f}%)"
            else:
                data_field = f"Rust {self.rust_counter}: Unknown (snapshot)"
            
            print(f"[CAM {self.camera_id}] ✅ Saved Rust ({save_type}): {filename}")
        
        elif mode == 'motion':
            filename = f"{base_path}/motion/images/motion_cam{self.camera_id}_{timestamp}.jpg"
            cv2.imwrite(filename, frame_to_save)
            self.motion_counter += 1
            
            if self.current_motion_data:
                total_rotation = self.current_motion_data.get('total_rotation', 0)
                direction = self.current_motion_data.get('direction', 'Unknown')
                rotation_deg = math.degrees(total_rotation)
                data_field = f"Motion {self.motion_counter}: {rotation_deg:.1f}° ({direction})"
            else:
                data_field = f"Motion {self.motion_counter}: No data (snapshot)"
            
            print(f"[CAM {self.camera_id}] ✅ Saved Motion ({save_type}): {filename}")
        
        else:
            # Generic snapshot
            filename = f"{base_path}/{mode}/images/{mode}_cam{self.camera_id}_{timestamp}.jpg"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            cv2.imwrite(filename, frame_to_save)
            data_field = f"{mode.capitalize()} snapshot"
            print(f"[CAM {self.camera_id}] ✅ Saved {mode} ({save_type}): {filename}")
        
        # ===== SAVE TO UNIFIED CSV =====
        csv_file = "ai/detections_unified.csv"
        file_exists = os.path.isfile(csv_file)
        
        row = {
            'timestamp': timestamp,
            'camera_id': f"cam{self.camera_id}",
            'detection_type': mode,
            'color': color_map.get(mode, 'gray'),
            'position': position_str,
            'data': str(data_field),
            'save_type': save_type
        }
        
        try:
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'camera_id', 'detection_type', 
                                                        'color', 'position', 'data', 'save_type'])
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            
            print(f"[CAM {self.camera_id}] ✅ Logged to CSV: {csv_file}")
            print(f"  → Type: {save_type}, Position: {position_str}")
            print(f"  → Data: {data_field}")
            
            # Unfreeze after save (if was frozen)
            if self.is_frozen:
                self.unfreeze()
            
            return True
        
        except Exception as e:
            print(f"[CAM {self.camera_id}] ❌ CSV save error: {e}")
            # return False
        
        # ===== START SAVE PROCESS =====
        import csv
        import datetime
        import os
        import math
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Color mapping for CSV
        color_map = {
            'landolt': 'yellow',
            'qr': 'blue',
            'hazmat': 'red',
            'crack': 'orange',
            'rust': 'brown',
            'motion': 'green'
        }
        
        # Camera offset distances
        CAMERA_OFFSETS = {0: 0.10, 1: 0.10, 2: 0.10}
        offset_distance = CAMERA_OFFSETS.get(self.camera_id, 0.10)
        
        def calculate_offset_position(drone_x, drone_y, drone_z, yaw_rad, offset_m):
            """Calculate detection position offset in front of drone."""
            if drone_x is None or drone_y is None or yaw_rad is None:
                return drone_x, drone_y, drone_z
            
            offset_x = drone_x + (offset_m * math.cos(yaw_rad))
            offset_y = drone_y + (offset_m * math.sin(yaw_rad))
            offset_z = drone_z if drone_z is not None else None
            return offset_x, offset_y, offset_z
        
        # Get drone telemetry data
        drone_x, drone_y, drone_z, drone_yaw = None, None, None, None
        try:
            record = self.drone_parser.get_latest()
            self.drone_parser.save_data()
            self.drone_parser.stop()
            
            if record:
                pos = self.drone_parser.get_position()
                rpy = self.drone_parser.get_rpy()
                
                if pos:
                    drone_x, drone_y, drone_z = pos['x'], pos['y'], pos['z']
                    print(f"[CAM {self.camera_id}] ✓ Position: X={drone_x:.2f}, Y={drone_y:.2f}, Z={drone_z:.2f}")
                
                if rpy:
                    drone_yaw = rpy['yaw']
                    print(f"[CAM {self.camera_id}] ✓ Yaw: {drone_yaw:.3f} rad ({rpy['yaw_deg']:.1f}°)")
            else:
                print(f"[CAM {self.camera_id}] ⚠️ No telemetry data")
        
        except Exception as e:
            print(f"[CAM {self.camera_id}] ⚠️ Telemetry error: {e}")
        
        # Calculate detection position
        det_x, det_y, det_z = calculate_offset_position(
            drone_x, drone_y, drone_z, drone_yaw, offset_distance
        )
        position_str = f"({det_x:.2f}, {det_y:.2f}, {det_z:.2f})" if det_x is not None else "(None, None, None)"
        
        # ===== MODE-SPECIFIC SAVE LOGIC =====
        data_field = "N/A"
        base_path = self.output_base
        
        if mode == 'qr':
            filename = f"{base_path}/qr/images/qr_cam{self.camera_id}_{timestamp}.jpg"
            cv2.imwrite(filename, frame_to_save)
            self.qr_counter += 1
            
            if self.current_qr_detections and len(self.current_qr_detections) > 0:
                qr_data = self.current_qr_detections[0]['qr_data']
                data_field = f"QRCode {self.qr_counter}: {qr_data}"
            else:
                data_field = f"QRCode {self.qr_counter}: No detection (manual save)"
            
            print(f"[CAM {self.camera_id}] Saved QR ({save_type}): {filename}")
        
        elif mode == 'landolt':
            filename = f"{base_path}/landolt/images/landolt_cam{self.camera_id}_{timestamp}.jpg"
            cv2.imwrite(filename, frame_to_save)
            self.landolt_counter += 1
            
            if self.current_landolt_results and len(self.current_landolt_results) > 0:
                best = max(self.current_landolt_results, key=lambda x: x['ring_confidence'])
                id_text = best.get('id_text', None)
                id_conf = best.get('id_confidence', 0)
                
                if id_text:
                    data_field = f"Landolt {self.landolt_counter}: ID {id_text} (Conf: {id_conf:.2f})"
                else:
                    data_field = f"Landolt {self.landolt_counter}: No ID"
            else:
                data_field = f"Landolt {self.landolt_counter}: No detection (manual save)"
            
            print(f"[CAM {self.camera_id}] Saved Landolt ({save_type}): {filename}")
        
        elif mode == 'hazmat':
            filename = f"{base_path}/hazmat/images/hazmat_cam{self.camera_id}_{timestamp}.jpg"
            cv2.imwrite(filename, frame_to_save)
            self.hazmat_counter += 1
            
            if self.current_hazmat_results and self.current_hazmat_results.boxes is not None:
                boxes = [box for box in self.current_hazmat_results.boxes 
                         if float(box.conf[0]) >= self.HAZMAT_CONFIDENCE_THRESHOLD]
                if boxes:
                    box = boxes[0]
                    cls_id = int(box.cls[0])
                    class_name = self.current_hazmat_results.names.get(cls_id, str(cls_id))
                    hazard = self.hazard_classes.get(class_name, class_name)
                    data_field = f"Hazmat {self.hazmat_counter}: {hazard}"
                else:
                    data_field = f"Hazmat {self.hazmat_counter}: No detection (manual save)"
            else:
                data_field = f"Hazmat {self.hazmat_counter}: No detection (manual save)"
            
            print(f"[CAM {self.camera_id}] Saved Hazmat ({save_type}): {filename}")
        
        elif mode == 'crack' and self.current_crack_analysis:
            capture_folder = f"{base_path}/crack/crack_capture_{timestamp}"
            os.makedirs(capture_folder, exist_ok=True)
            
            cv2.imwrite(f"{capture_folder}/crack_original_{timestamp}.jpg", 
                       self.current_crack_analysis['original_frame'])
            cv2.imwrite(f"{capture_folder}/crack_analysis_{timestamp}.jpg", 
                       self.current_crack_analysis['display'])
            
            self.crack_counter += 1
            
            if self.crack_results_data and len(self.crack_results_data) > 0:
                crack = self.crack_results_data[0]
                data_field = f"Crack {self.crack_counter}: {crack.length:.4f}m ({crack.classification})"
            else:
                data_field = f"Crack {self.crack_counter}: Unknown length"
            
            print(f"[CAM {self.camera_id}] Saved Crack ({save_type}): {capture_folder}")
        
        elif mode == 'rust' and self.current_rust_analysis:
            filename = f"{base_path}/rust/images/rust_cam{self.camera_id}_{timestamp}.jpg"
            cv2.imwrite(filename, frame_to_save)
            
            analysis_file = f"{base_path}/rust/images/rust_analysis_cam{self.camera_id}_{timestamp}.jpg"
            cv2.imwrite(analysis_file, self.current_rust_analysis['display'])
            
            self.rust_counter += 1
            
            if self.rust_results_data and len(self.rust_results_data) > 0:
                corrosion = self.rust_results_data[0]
                data_field = f"Rust {self.rust_counter}: {corrosion.dominant_class} ({corrosion.total_affected_percentage:.1f}%)"
            else:
                data_field = f"Rust {self.rust_counter}: Unknown degree"
            
            print(f"[CAM {self.camera_id}] Saved Rust ({save_type}): {filename}")
        
        elif mode == 'motion':
            filename = f"{base_path}/motion/images/motion_cam{self.camera_id}_{timestamp}.jpg"
            cv2.imwrite(filename, frame_to_save)
            self.motion_counter += 1
            
            if self.current_motion_data:
                total_rotation = self.current_motion_data.get('total_rotation', 0)
                direction = self.current_motion_data.get('direction', 'Unknown')
                rotation_deg = math.degrees(total_rotation)
                data_field = f"Motion {self.motion_counter}: {rotation_deg:.1f}° ({direction})"
            else:
                data_field = f"Motion {self.motion_counter}: No data (manual save)"
            
            print(f"[CAM {self.camera_id}] Saved Motion ({save_type}): {filename}")
        
        else:
            filename = f"{base_path}/unknown/images/unknown_cam{self.camera_id}_{timestamp}.jpg"
            cv2.imwrite(filename, frame_to_save)
            data_field = f"Unknown {mode}"
        
        # ===== SAVE TO UNIFIED CSV =====
        csv_file = "ai/detections_unified.csv"
        file_exists = os.path.isfile(csv_file)
        
        row = {
            'timestamp': timestamp,
            'camera_id': f"cam{self.camera_id}",
            'detection_type': mode,
            'color': color_map.get(mode, 'gray'),
            'position': position_str,
            'data': str(data_field),
            'save_type': save_type  # NEW: Track save type
        }
        
        try:
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'camera_id', 'detection_type', 
                                                        'color', 'position', 'data', 'save_type'])
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            
            print(f"[CAM {self.camera_id}] ✅ Logged to CSV: {csv_file}")
            print(f"  → Type: {save_type}, Position: {position_str}, Data: {data_field}")
            
            # Unfreeze after save (if was frozen)
            if self.is_frozen:
                self.unfreeze()
            
            return True
        
        except Exception as e:
            print(f"[CAM {self.camera_id}] ❌ CSV save error: {e}")
            return False

    
    def freeze(self, frame):
        self.is_frozen = True
        self.frozen_frame = frame.copy()
        print(f"[CAM {self.camera_id}] FROZEN - ready to save")
    
    def unfreeze(self):
        self.is_frozen = False
        if self.frozen_frame is not None:
            del self.frozen_frame
            self.frozen_frame = None
        print(f"[CAM {self.camera_id}] UNFROZEN")

# ============================================================================
# MULTI-CAMERA CONTROLLER
# ============================================================================

class MultiCameraController:
    def __init__(self, num_cameras=3, 
                 crack_weights='crack.pt',
                 hazmat_weights='hazmat.pt',
                 rust_model='deeplabv3_corrosion_multiclass.pth'):
        
        self.shared_models = SharedModelManager()
        self.shared_models.load_crack_model(crack_weights)
        self.shared_models.load_hazmat_model(hazmat_weights)
        self.shared_models.load_rust_model(rust_model)
        self.shared_models.load_ocr_reader()
        
        self.num_cameras = num_cameras
        self.workers = [CameraWorker(i, self.shared_models) for i in range(num_cameras)]
        
        self.MODES = {
            'STANDBY': 'standby',
            'QR': 'qr',
            'HAZMAT': 'hazmat',
            'CRACK': 'crack',
            'RUST': 'rust',
            'LANDOLT': 'landolt',
            'MOTION': 'motion'
        }
        
        self.current_mode = self.MODES['STANDBY']
        
        print(f"\n[OK] Multi-camera controller ready - {num_cameras} cameras")
        print("\nControls:")
        print("  Q - QR Detection (Enhanced 3D & Auto-Save)")
        print("  W - Hazmat Detection")
        print("  E - Crack Detection")
        print("  R - Rust Detection (FULL with Auto-Save & Position)")
        print("  T - Landolt Detection")
        print("  Y - Motion Detection")
        print("  J - Save Camera 0 (if frozen)")
        print("  K - Save Camera 1 (if frozen)")
        print("  L - Save Camera 2 (if frozen)")
        print("  P - Discard all & STANDBY")
        print("  ESC - Quit")
    
    def switch_mode(self, new_mode):
        if new_mode == self.current_mode:
            return
        
        self.current_mode = new_mode
        
        for worker in self.workers:
            worker.unfreeze()
        
        print(f"\n[MODE] -> {new_mode.upper()}")
    
    def process_frames(self, frame0, frame1, frame2):
        frames = [frame0, frame1, frame2]
        processed_frames = []
        
        for i, (frame, worker) in enumerate(zip(frames, self.workers)):
            # ✅ STORE RAW FRAME FIRST (before any processing)
            if frame is not None:
                worker.raw_frame = frame.copy()
            
            if worker.is_frozen:
                display = worker.frozen_frame.copy()
                cv2.putText(display, f"CAM {i} FROZEN", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display, f"Press {'J' if i==0 else 'K' if i==1 else 'L'} to save", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                processed_frames.append(display)
                continue
            
            annotated = frame.copy()
            should_freeze = False
            
            try:
                if self.current_mode == self.MODES['STANDBY']:
                    cv2.putText(annotated, f"CAM {i} - STANDBY", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                elif self.current_mode == self.MODES['LANDOLT']:
                    results = worker.process_landolt(frame)
                    annotated = worker.annotate_landolt(frame, results)
                    should_freeze = len(results) > 0
                    
                    status = f"CAM {i} - LANDOLT ({len(results)})"
                    if worker.save_counter > 0:
                        status += f" | Saved: {worker.save_counter}"
                    cv2.putText(annotated, status, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                elif self.current_mode == self.MODES['QR']:
                    detections = worker.process_qr(frame)
                    annotated = worker.annotate_qr(frame, detections)
                    should_freeze = len(detections) > 0
                    
                    status = f"CAM {i} - QR ({len(detections)})"
                    if worker.qr_detector_3d.qr_save_counter > 0:
                        status += f" | Saved: {worker.qr_detector_3d.qr_save_counter}"
                    cv2.putText(annotated, status, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                elif self.current_mode == self.MODES['HAZMAT']:
                    results = worker.process_hazmat(frame)
                    if results and results.boxes is not None:
                        filtered_boxes = [box for box in results.boxes if float(box.conf[0]) >= worker.HAZMAT_CONFIDENCE_THRESHOLD]
                        should_freeze = len(filtered_boxes) > 0
                        
                        for box in filtered_boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            class_name = results.names[cls_id] if hasattr(results, 'names') else f"class_{cls_id}"
                            hazard_label = worker.hazard_classes.get(class_name, class_name)
                            
                            distance_x, position_y, height_z, _, _, _ = worker._calculate_hazmat_real_world_coordinates(box)
                            
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(annotated, f"{hazard_label}",
                                       (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(annotated, f"X:{distance_x:.2f}m Y:{position_y:+.2f}m Z:{height_z:.2f}m",
                                       (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                            cv2.putText(annotated, f"Conf: {conf:.2f}",
                                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    status = f"CAM {i} - HAZMAT"
                    if len(worker.hazmat_saved_objects) > 0:
                        status += f" | Saved: {len(worker.hazmat_saved_objects)}"
                    cv2.putText(annotated, status, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                elif self.current_mode == self.MODES['CRACK']:
                    analysis = worker.process_crack(frame)
                    should_freeze = analysis is not None
                    
                    if analysis:
                        cv2.drawContours(annotated, [analysis['square']], -1, (0, 255, 0), 2)
                    
                    cv2.putText(annotated, f"CAM {i} - CRACK", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                elif self.current_mode == self.MODES['RUST']:
                    analysis = worker.process_rust(frame)
                    should_freeze = analysis is not None
                    
                    if analysis:
                        cv2.drawContours(annotated, [analysis['square']], -1, (0, 255, 0), 2)
                        
                        if worker.rust_position_data:
                            pos = worker.rust_position_data
                            info_y = 50
                            cv2.putText(annotated, f"X: {pos.x_distance_m:.2f}m", 
                                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            cv2.putText(annotated, f"Y: {pos.y_distance_m:+.2f}m", 
                                       (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            cv2.putText(annotated, f"Z: {pos.z_height_ground_m:.2f}m", 
                                       (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        if worker.rust_results_data:
                            corr = worker.rust_results_data[0]
                            info_y = 125
                            cv2.putText(annotated, f"Severity: {corr.total_affected_percentage:.1f}%", 
                                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            cv2.putText(annotated, f"Class: {corr.dominant_class}", 
                                       (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            if corr.is_critical:
                                cv2.putText(annotated, "CRITICAL!", 
                                           (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    status = f"CAM {i} - RUST"
                    if len(worker.rust_saved_objects) > 0:
                        status += f" | Saved: {len(worker.rust_saved_objects)}"
                    cv2.putText(annotated, status, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                elif self.current_mode == self.MODES['MOTION']:
                    worker.process_motion(frame)
                    cv2.putText(annotated, f"CAM {i} - MOTION", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # ✅ STORE PROCESSED/ANNOTATED FRAME
                worker.last_processed_frame = annotated.copy()
                
                if should_freeze and not worker.is_frozen:
                    worker.freeze(annotated)
            
            except Exception as e:
                logger.error(f"[CAM {i}] Processing error in {self.current_mode}: {e}")
                cv2.putText(annotated, f"CAM {i} - ERROR", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # ✅ EVEN ON ERROR, STORE FRAME
                worker.last_processed_frame = annotated.copy()
            
            processed_frames.append(annotated)
        
        return processed_frames
    
    def handle_key(self, key):
        if key == ord('q'):
            self.switch_mode(self.MODES['QR'])
        elif key == ord('w'):
            self.switch_mode(self.MODES['HAZMAT'])
        elif key == ord('e'):
            self.switch_mode(self.MODES['CRACK'])
        elif key == ord('r'):
            self.switch_mode(self.MODES['RUST'])
        elif key == ord('t'):
            self.switch_mode(self.MODES['LANDOLT'])
        elif key == ord('y'):
            self.switch_mode(self.MODES['MOTION'])
        elif key == ord('j'):
            # self.workers[0].save_current_detection(self.current_mode)
            if self.workers[0].is_frozen:
                self.workers[0].save_current_detection(self.current_mode)
            else:
                 print("[CAM 0] Not frozen - cannot save")
        elif key == ord('k'):
            if (len(self.workers) > 1 and self.workers[1].is_frozen):
                self.workers[1].save_current_detection(self.current_mode)
            else:
                print("[CAM 1] Not frozen - cannot save")
        elif key == ord('l'):
            if (len(self.workers) > 2 and self.workers[2].is_frozen):
                self.workers[2].save_current_detection(self.current_mode)
            else:
                print("[CAM 2] Not frozen - cannot save")
        elif key == ord('p'):
            for worker in self.workers:
                worker.unfreeze()
            self.switch_mode(self.MODES['STANDBY'])
            print("[ALL] Discarded - returned to STANDBY")
        elif key == 27:
            return 'quit'
        
        return 'continue'
    
    def cleanup(self):
        for worker in self.workers:
            worker.unfreeze()
        gpu_manager.cleanup_memory()

# ============================================================================
# MAIN
# ============================================================================

def create_black_frame(width=640, height=480):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, "CAMERA FAILED", (width//2 - 100, height//2), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return frame

def main():
    camera_indices = [0, 1, 2]
    
    controller = MultiCameraController(
        num_cameras=3,
        crack_weights='crack.pt',
        hazmat_weights='hazmat.pt',
        rust_model='deeplabv3_corrosion_multiclass.pth'
    )
    
    cameras = []
    for idx in camera_indices:
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cameras.append(cap)
    
    failed_cameras = []
    for i, cap in enumerate(cameras):
        if not cap.isOpened():
            print(f"[WARNING] Camera {camera_indices[i]} failed to open")
            failed_cameras.append(i)
    
    if len(failed_cameras) == len(cameras):
        print("[ERROR] All cameras failed to open")
        return 1
    
    print("\n" + "="*70)
    print("MULTI-CAMERA DETECTION SYSTEM - FULL RUST DETECTION")
    print("="*70)
    print("Press R for FULL Rust Detection with Auto-Save & Position Analysis")
    print("="*70 + "\n")
    
    try:
        while True:
            frames = []
            for i, cap in enumerate(cameras):
                if i in failed_cameras:
                    frames.append(create_black_frame())
                    continue
                
                try:
                    ret, frame = cap.read()
                    if not ret:
                        frames.append(create_black_frame())
                    else:
                        frames.append(frame)
                except Exception as e:
                    logger.error(f"[ERROR] Camera {camera_indices[i]} exception: {e}")
                    frames.append(create_black_frame())
            
            while len(frames) < 3:
                frames.append(create_black_frame())
            
            try:
                processed_frames = controller.process_frames(frames[0], frames[1], frames[2])
            except Exception as e:
                logger.error(f"[ERROR] Processing exception: {e}")
                processed_frames = frames
            
            try:
                combined = np.hstack(processed_frames)
            except Exception as e:
                logger.error(f"[ERROR] Stacking exception: {e}")
                h, w = 480, 640
                resized = [cv2.resize(f, (w, h)) for f in processed_frames]
                combined = np.hstack(resized)
            
            cv2.imshow("Multi-Camera [Q=QR R=RUST T=Landolt J/K/L=Save P=Discard ESC=Quit]", combined)
            
            key = cv2.waitKey(1) & 0xFF
            action = controller.handle_key(key)
            
            if action == 'quit':
                break
    
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user")
    
    except Exception as e:
        logger.error(f"[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n[CLEANUP] Shutting down...")
        for cap in cameras:
            if cap.isOpened():
                cap.release()
        cv2.destroyAllWindows()
        controller.cleanup()
        print("[OK] System shutdown complete")
    
    return 0

if __name__ == "__main__":
    exit(main())