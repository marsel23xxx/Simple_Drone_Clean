"""
AI Detection Controller - Full Implementation
Main controller untuk semua AI detectors
"""

import cv2
import time
import math
import numpy as np
import torch
from typing import Tuple, Optional
import logging

from .gpu_manager import gpu_manager
from .detectors import (
    CrackDetectorLibrary,
    HazmatDetector, 
    QRDetector,
    LandoltDetector,
    MotionDetector,
    RustDetector
)

logger = logging.getLogger(__name__)


class AIDetectionController:
    """
    Main controller untuk AI detection system
    Support external camera feed
    """
    
    def __init__(
        self,
        crack_weights='models/crack.pt',
        hazmat_weights='models/hazmat.pt',
        rust_model_path='models/deeplabv3_corrosion_multiclass.pth',
        camera_index=0,
        external_feed=False
    ):
        self.external_feed = external_feed
        self.camera_index = camera_index
        self.cap = None
        
        self.gpu_manager = gpu_manager
        
        self.MODES = {
            'STANDBY': 'standby',
            'CRACK': 'crack_detection',
            'HAZMAT': 'hazmat_detection',
            'QR': 'qr_detection',
            'RUST': 'rust_detection',
            'MOTION': 'motion_detection',
            'LANDOLT': 'landolt_detection'
        }
        
        self.current_mode = self.MODES['STANDBY']
        self.previous_mode = self.MODES['STANDBY']
        
        self.crack_detector = None
        self.hazmat_detector = None
        self.qr_detector = None
        self.rust_detector = None
        self.motion_detector = None
        self.landolt_detector = None
        
        self.crack_weights = crack_weights
        self.hazmat_weights = hazmat_weights
        self.rust_model_path = rust_model_path
        
        self.crack_prescreener = None
        self.hazmat_prescreener = None
        
        self.detection_history = []
        self.history_size = 10
        self.mode_switch_threshold = 3
        self.no_detection_threshold = 30
        self.no_detection_count = 0
        
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        
        self.gpu_warmup_done = False
        self.gpu_cleanup_interval = 60
        
        print("AI Detection Controller initialized")
        print(f"GPU: {'ENABLED' if self.gpu_manager.cuda_available else 'DISABLED'}")
        print(f"External feed mode: {external_feed}")
    
    def start_services(self):
        """Start detection services"""
        if not self.external_feed:
            return self.initialize_camera()
        else:
            return self.initialize_prescreeners()
    
    def initialize_camera(self):
        """Initialize internal camera"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_index}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Camera {self.camera_index} initialized")
        return True
    
    def initialize_prescreeners(self):
        """Initialize prescreening models"""
        try:
            from ultralytics import YOLO
            
            self.crack_prescreener = YOLO(self.crack_weights)
            self.hazmat_prescreener = YOLO(self.hazmat_weights)
            
            self.crack_prescreener.overrides['conf'] = 0.3
            self.hazmat_prescreener.overrides['conf'] = 0.3
            
            print("Prescreening models loaded")
            return True
        except Exception as e:
            print(f"Error loading prescreeners: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Process frame untuk detection"""
        self.frame_count += 1
        
        current_time = time.time()
        time_diff = current_time - self.last_fps_time
        if time_diff >= 1.0:
            self.fps = self.frame_count / time_diff
            self.frame_count = 0
            self.last_fps_time = current_time
        
        if not self.gpu_warmup_done:
            self.gpu_warmup()
        
        has_hazmat, has_qr, has_landolt, has_motion, has_rust, has_crack = \
            self.prescreen_frame(frame)
        
        self.update_detection_history(
            has_hazmat, has_qr, has_landolt, 
            has_motion, has_rust, has_crack
        )
        
        processed_frame = frame.copy()
        analysis_frame = None
        
        if self.current_mode == self.MODES['CRACK'] and self.crack_detector:
            processed_frame, _ = self.crack_detector.process_frame(frame)
            
        elif self.current_mode == self.MODES['HAZMAT'] and self.hazmat_detector:
            try:
                with torch.no_grad():
                    results = self.hazmat_detector.model(frame, device=self.gpu_manager.device)[0]
                processed_frame = self.hazmat_detector.annotate_frame(frame, results)
            except Exception as e:
                logger.warning(f"Hazmat error: {e}")
                processed_frame = frame.copy()
            
        elif self.current_mode == self.MODES['QR'] and self.qr_detector:
            try:
                qr_detections = self.qr_detector.process_frame(frame)
                processed_frame = self.qr_detector.annotate_frame(frame, qr_detections)
            except Exception as e:
                logger.warning(f"QR error: {e}")
                processed_frame = frame.copy()
            
        elif self.current_mode == self.MODES['LANDOLT'] and self.landolt_detector:
            try:
                landolt_results = self.landolt_detector.process_frame(frame)
                processed_frame = self.landolt_detector.annotate_frame(frame, landolt_results)
            except Exception as e:
                logger.warning(f"Landolt error: {e}")
                processed_frame = frame.copy()
            
        elif self.current_mode == self.MODES['MOTION'] and self.motion_detector:
            try:
                motion_results = self.motion_detector.process_frame(frame)
                processed_frame = self.motion_detector.annotate_frame(frame, motion_results)
            except Exception as e:
                logger.warning(f"Motion error: {e}")
                processed_frame = frame.copy()
            
        elif self.current_mode == self.MODES['RUST'] and self.rust_detector:
            try:
                result = self.rust_detector.process_frame(frame)
                processed_frame = result['main_frame']
                analysis_frame = result['analysis_frame']
            except Exception as e:
                logger.warning(f"Rust error: {e}")
                processed_frame = frame.copy()
                analysis_frame = None
            
        else:
            processed_frame = self.draw_standby_overlay(
                frame, has_hazmat, has_qr, has_landolt, 
                has_motion, has_rust, has_crack
            )
        
        self.add_ai_overlay(processed_frame)
        
        if self.frame_count % self.gpu_cleanup_interval == 0:
            self.gpu_manager.cleanup_memory()
        
        return processed_frame, analysis_frame
    
    def prescreen_frame(self, frame: np.ndarray) -> Tuple[bool, bool, bool, bool, bool, bool]:
        """Prescreen frame untuk deteksi cepat"""
        small_frame = cv2.resize(frame, (320, 240))
        
        has_hazmat = False
        has_qr = False
        has_landolt = False
        has_motion = False
        has_rust = False
        has_crack = False
        
        try:
            if self.hazmat_prescreener:
                with torch.no_grad():
                    results = self.hazmat_prescreener(small_frame, verbose=False, device=self.gpu_manager.device)
                has_hazmat = len(results[0].boxes) > 0 if results[0].boxes is not None else False
            
            if not has_hazmat:
                qr_detector = cv2.QRCodeDetector()
                data, bbox, _ = qr_detector.detectAndDecode(small_frame)
                has_qr = bool(data and bbox is not None)
            
            if not has_hazmat and not has_qr:
                has_landolt = self._quick_landolt_detection(small_frame)
            
            if not has_hazmat and not has_qr and not has_landolt:
                has_motion = self._quick_motion_detection(small_frame)
            
            if not has_hazmat and not has_qr and not has_landolt and not has_motion:
                has_rust = self._quick_square_detection(small_frame)
            
            if (not has_hazmat and not has_qr and not has_landolt and 
                not has_motion and not has_rust and self.crack_prescreener):
                with torch.no_grad():
                    results = self.crack_prescreener(small_frame, verbose=False, device=self.gpu_manager.device)
                has_crack = len(results[0].boxes) > 0 if results[0].boxes is not None else False
            
        except Exception as e:
            logger.warning(f"Prescreening error: {e}")
        
        return has_hazmat, has_qr, has_landolt, has_motion, has_rust, has_crack
    
    def _quick_landolt_detection(self, frame):
        """Quick Landolt detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 2)
            edges = cv2.Canny(blurred, 60, 180)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 50 or area > 8000:
                    continue
                    
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                    
                circularity = 4 * math.pi * area / (perimeter * perimeter)
                if circularity > 0.65:
                    return True
            return False
        except:
            return False
    
    def _quick_motion_detection(self, frame):
        """Quick motion detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 2)
            edges = cv2.Canny(blurred, 0, 122)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 30:
                    continue
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * math.pi * area / (perimeter * perimeter)
                if circularity > 0.7:
                    return True
            return False
        except:
            return False
    
    def _quick_square_detection(self, frame):
        """Quick square detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cv2.contourArea(cnt) > 1000:
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                    if len(approx) == 4:
                        return True
            return False
        except:
            return False
    
    def update_detection_history(self, has_hazmat, has_qr, has_landolt, 
                                 has_motion, has_rust, has_crack):
        """Update detection history"""
        if has_hazmat:
            detection_type = self.MODES['HAZMAT']
        elif has_qr:
            detection_type = self.MODES['QR']
        elif has_landolt:
            detection_type = self.MODES['LANDOLT']
        elif has_motion:
            detection_type = self.MODES['MOTION']
        elif has_rust:
            detection_type = self.MODES['RUST']
        elif has_crack:
            detection_type = self.MODES['CRACK']
        else:
            detection_type = None
        
        self.detection_history.append(detection_type)
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
        
        recent = self.detection_history[-self.mode_switch_threshold:]
        
        for mode in [self.MODES['HAZMAT'], self.MODES['QR'], self.MODES['LANDOLT'],
                     self.MODES['MOTION'], self.MODES['RUST'], self.MODES['CRACK']]:
            if recent.count(mode) >= self.mode_switch_threshold:
                if self.current_mode != mode:
                    self.switch_mode(mode)
                self.no_detection_count = 0
                return
        
        if recent.count(None) >= self.mode_switch_threshold:
            self.no_detection_count += 1
            if self.no_detection_count > self.no_detection_threshold:
                if self.current_mode != self.MODES['STANDBY']:
                    self.switch_mode(self.MODES['STANDBY'])
    
    def switch_mode(self, new_mode: str):
        """Switch detection mode"""
        self.previous_mode = self.current_mode
        self.current_mode = new_mode
        
        print(f"Mode Switch: {self.previous_mode} -> {self.current_mode}")
        
        if new_mode == self.MODES['CRACK']:
            self.initialize_crack_detector()
        elif new_mode == self.MODES['HAZMAT']:
            self.initialize_hazmat_detector()
        elif new_mode == self.MODES['QR']:
            self.initialize_qr_detector()
        elif new_mode == self.MODES['LANDOLT']:
            self.initialize_landolt_detector()
        elif new_mode == self.MODES['RUST']:
            self.initialize_rust_detector()
        elif new_mode == self.MODES['MOTION']:
            self.initialize_motion_detector()
        
        self.no_detection_count = 0
        self.detection_history.clear()
        
        if self.gpu_manager.cuda_available:
            self.gpu_manager.cleanup_memory()
    
    def initialize_crack_detector(self):
        """Initialize crack detector"""
        if self.crack_detector is None:
            try:
                self.crack_detector = CrackDetectorLibrary(
                    weights=self.crack_weights,
                    output_dir='ai_crack_captures'
                )
                print("Crack detector activated")
            except Exception as e:
                print(f"Error initializing crack detector: {e}")
    
    def initialize_hazmat_detector(self):
        """Initialize hazmat detector"""
        if self.hazmat_detector is None:
            try:
                self.hazmat_detector = HazmatDetector(self.hazmat_weights)
                print("Hazmat detector activated")
            except Exception as e:
                print(f"Error initializing hazmat detector: {e}")
    
    def initialize_qr_detector(self):
        """Initialize QR detector"""
        if self.qr_detector is None:
            try:
                self.qr_detector = QRDetector(output_dir='ai_qr_captures')
                print("QR detector activated")
            except Exception as e:
                print(f"Error initializing QR detector: {e}")
    
    def initialize_landolt_detector(self):
        """Initialize Landolt detector"""
        if self.landolt_detector is None:
            try:
                self.landolt_detector = LandoltDetector(output_dir='ai_landolt_captures')
                print("Landolt detector activated")
            except Exception as e:
                print(f"Error initializing Landolt detector: {e}")
    
    def initialize_motion_detector(self):
        """Initialize motion detector"""
        if self.motion_detector is None:
            try:
                self.motion_detector = MotionDetector(output_dir='ai_motion_captures')
                print("Motion detector activated")
            except Exception as e:
                print(f"Error initializing motion detector: {e}")
    
    def initialize_rust_detector(self):
        """Initialize rust detector"""
        if self.rust_detector is None:
            try:
                self.rust_detector = RustDetector(
                    model_path=self.rust_model_path,
                    save_folder='ai_rust_captures'
                )
                print("Rust detector activated")
            except Exception as e:
                print(f"Error initializing rust detector: {e}")
    
    def gpu_warmup(self):
        """GPU warmup"""
        if self.gpu_manager.cuda_available and not self.gpu_warmup_done:
            try:
                dummy = torch.randn(1, 3, 224, 224, device=self.gpu_manager.device)
                _ = torch.sum(dummy)
                del dummy
                torch.cuda.empty_cache()
                self.gpu_warmup_done = True
                print("[GPU] Warmup completed")
            except:
                pass
    
    def draw_standby_overlay(self, frame, has_hazmat, has_qr, has_landolt,
                            has_motion, has_rust, has_crack):
        """Draw standby overlay"""
        overlay = frame.copy()
        
        gpu_status = "GPU ON" if self.gpu_manager.cuda_available else "GPU OFF"
        cv2.putText(overlay, f"AI DETECTION - SCANNING [{gpu_status}]", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        if has_hazmat:
            cv2.putText(overlay, "HAZMAT DETECTED", 
                       (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        elif has_qr:
            cv2.putText(overlay, "QR CODE DETECTED", 
                       (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        elif has_landolt:
            cv2.putText(overlay, "LANDOLT RING DETECTED", 
                       (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        elif has_motion:
            cv2.putText(overlay, "MOTION DETECTED", 
                       (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        elif has_rust:
            cv2.putText(overlay, "RUST DETECTED", 
                       (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 0), 1)
        elif has_crack:
            cv2.putText(overlay, "CRACK DETECTED", 
                       (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.putText(overlay, "CLEAR", 
                       (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return overlay
    
    def add_ai_overlay(self, frame):
        """Add AI info overlay"""
        height, width = frame.shape[:2]
        
        mode_colors = {
            self.MODES['STANDBY']: (128, 128, 128),
            self.MODES['CRACK']: (0, 255, 0),
            self.MODES['HAZMAT']: (0, 0, 255),
            self.MODES['QR']: (255, 0, 255),
            self.MODES['LANDOLT']: (0, 255, 0),
            self.MODES['MOTION']: (255, 255, 0),
            self.MODES['RUST']: (255, 128, 0)
        }
        
        mode_color = mode_colors.get(self.current_mode, (255, 255, 255))
        
        gpu_indicator = "GPU" if self.gpu_manager.cuda_available else "CPU"
        if self.current_mode == self.MODES['RUST']:
            mode_text = f"RUST+ANALYSIS {gpu_indicator}"
        else:
            mode_text = f"{self.current_mode.upper()[:8]} {gpu_indicator}"
        
        text_x = max(width - 160, 10)
        cv2.putText(frame, mode_text, (text_x, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, mode_color, 1)
        
        crack_count = self.crack_detector.get_capture_count() if self.crack_detector else 0
        hazmat_count = len(self.hazmat_detector.saved_objects) if self.hazmat_detector else 0
        qr_count = len(self.qr_detector.saved_qr_codes) if self.qr_detector else 0
        landolt_count = self.landolt_detector.get_capture_count() if self.landolt_detector else 0
        motion_count = self.motion_detector.get_capture_count() if self.motion_detector else 0
        rust_count = self.rust_detector.get_capture_count() if self.rust_detector else 0
        
        stats_x = max(width - 240, 10)
        stats_y = max(height - 10, 30)
        cv2.putText(frame, f"C:{crack_count} H:{hazmat_count} Q:{qr_count} L:{landolt_count} M:{motion_count} R:{rust_count}", 
                   (stats_x, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        if self.current_mode == self.MODES['RUST']:
            cv2.putText(frame, f"Analysis: ACTIVE | GPU: {'ON' if self.gpu_manager.cuda_available else 'OFF'}", 
                       (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        else:
            cv2.putText(frame, f"Analysis: OFF | GPU: {'ON' if self.gpu_manager.cuda_available else 'OFF'}", 
                       (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        
        if self.gpu_manager.cuda_available:
            self.gpu_manager.cleanup_memory()
        
        print("AI Detection Controller stopped")


__all__ = ['AIDetectionController']