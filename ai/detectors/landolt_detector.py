"""
Landolt Ring Detection Module  
GPU-optimized Landolt ring detection with OCR
"""

import cv2
import numpy as np
import math
import time
import datetime
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from ..gpu_manager import gpu_manager

# Try to import easyocr
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[Landolt] EasyOCR not available. OCR will be disabled.")

# Pattern for Landolt Ring ID
ID_PATTERN = re.compile(r"^[0-9]+[A-Z]?$")


@dataclass
class LandoltData:
    ring_x: int
    ring_y: int
    ring_r: int
    ring_confidence: float
    ring_gap_ratio: float
    id_text: Optional[str]
    id_confidence: float
    frames_tracked: int
    stable: bool
    x_distance: float
    y_lateral: float
    z_height: float
    direction: str
    
    @property
    def is_critical(self) -> bool:
        return not self.stable or self.ring_confidence < 0.6
    
    @property
    def classification(self) -> str:
        if self.id_text:
            return f"ID_{self.id_text}"
        else:
            return "Unknown_ID"


class CameraAnalytics:
    def __init__(self, frame_width=640, frame_height=480):
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
        
        self.device = gpu_manager.device
        
    def update_frame_dimensions(self, width, height):
        self.frame_width = width
        self.frame_height = height
        self.frame_center_x = width // 2
        self.frame_center_y = height // 2
        
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


class LandoltDetector:
    def __init__(self, output_dir='ai_landolt_captures', confidence_threshold=0.35):
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        
        # Create folders
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "analytics"), exist_ok=True)
        
        # Initialize EasyOCR
        self.reader = None
        if EASYOCR_AVAILABLE:
            self.reader = self._make_reader()
        
        # Camera Analytics
        self.camera_analytics = CameraAnalytics()
        
        # Detection parameters
        self.dp = 1.0
        self.min_dist = 60
        self.param1 = 80
        self.param2 = 25
        self.min_radius = 25
        self.max_radius = 120
        
        # Tracking
        self.tracked_rings = []
        self.max_tracking_history = 5
        self.position_smoothing = 0.3
        self.tracking_distance_threshold = 40
        self.min_tracking_frames = 3
        
        # OCR parameters
        self.ocr_min_conf = 0.4
        self.ocr_expand_factor = 1.2
        self.ocr_cache = {}
        self.ocr_cache_timeout = 10
        
        # Auto save
        self.auto_save_enabled = True
        self.saved_positions = set()
        self.min_stable_frames = 3
        self.min_ring_confidence = 0.4
        self.position_tolerance = 30
        self.save_counter = 0
        
        # Detection tracking
        self.detection_count = {}
        self.saved_landolt_objects = set()
        self.STABILIZATION_FRAMES = 3
        
        # GPU
        self.device = gpu_manager.device
        
        print(f"[Landolt] Detector initialized with GPU: {gpu_manager.cuda_available}")
    
    def _make_reader(self):
        try:
            gpu_ok = gpu_manager.cuda_available
            reader = easyocr.Reader(['en'], gpu=gpu_ok, verbose=False)
            return reader
        except Exception as e:
            print(f"[Landolt] EasyOCR setup failed: {e}")
            return easyocr.Reader(['en'], gpu=False, verbose=False)
    
    def update_frame_dimensions(self, width, height):
        self.camera_analytics.update_frame_dimensions(width, height)
    
    def process_frame(self, frame: np.ndarray) -> List[LandoltData]:
        """Process frame for Landolt ring detection"""
        frame_height, frame_width = frame.shape[:2]
        self.update_frame_dimensions(frame_width, frame_height)
        
        # Detect rings
        raw_detections = self.detect_rings_raw(frame)
        
        # Update tracking
        frame_count = getattr(self, '_frame_count', 0)
        self._frame_count = frame_count + 1
        self.update_tracking(raw_detections, frame_count)
        
        combined_results = []
        
        for ring in self.tracked_rings:
            if ring['stable'] or ring['confidence'] > 0.5:
                x, y, r = int(ring['x']), int(ring['y']), int(ring['r'])
                
                # OCR
                detected_ids = []
                if not ring['stable'] or ring['frames_tracked'] % 5 == 0:
                    detected_ids = self.ocr_around_ring_cached(frame, x, y, r, frame_count)
                    ring['last_ocr_ids'] = detected_ids
                else:
                    detected_ids = ring.get('last_ocr_ids', [])
                
                best_id = None
                if detected_ids:
                    best_id = max(detected_ids, key=lambda x: x['confidence'])
                
                # Calculate position
                x_distance, y_lateral, z_height, direction = \
                    self.camera_analytics.pixels_to_3d_coordinates(x, y, r)
                
                result = LandoltData(
                    ring_x=x,
                    ring_y=y,
                    ring_r=r,
                    ring_confidence=ring['confidence'],
                    ring_gap_ratio=ring['gap_ratio'],
                    id_text=best_id['id'] if best_id else None,
                    id_confidence=best_id['confidence'] if best_id else 0,
                    frames_tracked=ring['frames_tracked'],
                    stable=ring['stable'],
                    x_distance=x_distance,
                    y_lateral=y_lateral,
                    z_height=z_height,
                    direction=direction
                )
                
                combined_results.append(result)
        
        return combined_results
    
    def detect_rings_raw(self, frame):
        """Raw ring detection"""
        processed, edges = self.preprocess_for_ring(frame)
        
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
                
                # Check structure
                best_confidence = 0
                
                roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
                _, binary = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) >= 2:
                    has_gap, gap_ratio = self.detect_gap_flexible(roi, roi_center, r/2, r/2)
                    
                    if has_gap:
                        confidence = gap_ratio * 2
                        if confidence > best_confidence and confidence > self.confidence_threshold:
                            best_confidence = confidence
                            detected_rings.append((x, y, r, gap_ratio, confidence, r*2, r*2))
        
        return detected_rings
    
    def preprocess_for_ring(self, frame):
        """Preprocess for ring detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        processed = cv2.GaussianBlur(bilateral, (3,3), 1.0)
        edges = cv2.Canny(processed, 60, 180)
        return processed, edges
    
    def detect_gap_flexible(self, roi, center, outer_r, inner_r):
        """Detect gap in ring"""
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
        
        if (0.08 <= gap_ratio <= 0.4 and max_gap_length >= 3 and len(gap_segments) <= 3):
            return True, gap_ratio
        
        return False, 0
    
    def update_tracking(self, current_detections, frame_count):
        """Update tracking"""
        updated_tracked = []
        
        for tracked in self.tracked_rings:
            matched = False
            best_match = None
            
            if current_detections:
                detection_points = np.array([(det[0], det[1]) for det in current_detections])
                tracked_point = np.array([tracked['x'], tracked['y']])
                distances = np.linalg.norm(detection_points - tracked_point, axis=1)
                
                valid_indices = distances < self.tracking_distance_threshold
                if np.any(valid_indices):
                    min_idx = np.argmin(distances[valid_indices])
                    valid_idx = np.where(valid_indices)[0][min_idx]
                    best_match = current_detections[valid_idx]
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
        
        # Add new
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
        
        self.tracked_rings = [ring for ring in updated_tracked 
                             if ring['frames_tracked'] >= self.min_tracking_frames or ring['confidence'] > 0.6]
        
        if len(self.tracked_rings) > 3:
            self.tracked_rings.sort(key=lambda x: x['confidence'], reverse=True)
            self.tracked_rings = self.tracked_rings[:3]
    
    def ocr_around_ring_cached(self, frame, x, y, r, frame_count):
        """OCR with caching"""
        if not self.reader:
            return []
        
        expand_r = int(r * self.ocr_expand_factor)
        x1 = max(0, x - expand_r)
        y1 = max(0, y - expand_r)
        x2 = min(frame.shape[1], x + expand_r)
        y2 = min(frame.shape[0], y + expand_r)
        
        ocr_roi = frame[y1:y2, x1:x2]
        
        if ocr_roi.size == 0:
            return []
        
        detected_ids = []
        variants = self.preprocess_for_ocr_simple(ocr_roi)
        
        results_total = []
        for var in variants:
            try:
                results = self.reader.readtext(var, detail=1, paragraph=False, width_ths=0.7)
                results_total.extend(results)
            except:
                continue
        
        seen = set()
        for box, text, conf in results_total:
            alnum = re.sub(r'[^A-Za-z0-9]', '', text).upper()
            if not alnum or conf < self.ocr_min_conf:
                continue
            if not ID_PATTERN.match(alnum):
                continue
            if alnum in seen:
                continue
            seen.add(alnum)
            
            detected_ids.append({'id': alnum, 'confidence': conf})
        
        return detected_ids
    
    def preprocess_for_ocr_simple(self, img):
        """Preprocess for OCR"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        th = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        return [
            cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(th, cv2.COLOR_GRAY2BGR),
        ]
    
    def annotate_frame(self, frame, landolt_results):
        """Annotate frame"""
        annotated = frame.copy()
        
        if not landolt_results:
            cv2.putText(annotated, "SCANNING FOR LANDOLT RINGS", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return annotated
        
        for result in landolt_results:
            x, y, r = result.ring_x, result.ring_y, result.ring_r
            
            if result.id_text:
                color = (0, 255, 0) if result.stable else (0, 255, 255)
            else:
                color = (0, 165, 255)
            
            cv2.circle(annotated, (x, y), r, color, 3)
            cv2.circle(annotated, (x, y), 3, (255, 0, 0), -1)
            
            status = "[STABLE]" if result.stable else f"[{result.frames_tracked}]"
            cv2.putText(annotated, status, (x + r - 35, y - r + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            if result.id_text:
                cv2.putText(annotated, f"ID: {result.id_text}", 
                           (x - 30, y + r + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return annotated
    
    def get_capture_count(self) -> int:
        return self.save_counter