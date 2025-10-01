"""
AI Detection System - Main controller for all AI detections
Integrates with PyQt5 UI through signals
"""

import cv2
import numpy as np
import time
import threading
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap

from .gpu_manager import gpu_manager
# Import detectors akan ditambahkan sesuai kebutuhan
# from .detectors.crack_detector import CrackDetector
# from .detectors.hazmat_detector import HazmatDetector
# dll


class AIDetectionSystem(QObject):
    """
    AI Detection System with PyQt5 integration
    Manages all detection modes and emits signals for UI updates
    """
    
    # Signals for UI updates
    detection_frame_ready = pyqtSignal(np.ndarray)  # Live detection frame
    capture_frame_ready = pyqtSignal(np.ndarray)    # Captured image
    detection_status = pyqtSignal(str)              # Status message
    mode_changed = pyqtSignal(str)                  # Current detection mode
    
    MODES = {
        'STANDBY': 'standby',
        'CRACK': 'crack_detection',
        'HAZMAT': 'hazmat_detection',
        'QR': 'qr_detection',
        'RUST': 'rust_detection',
        'MOTION': 'motion_detection',
        'LANDOLT': 'landolt_detection'
    }
    
    def __init__(self, crack_weights='models/crack.pt', 
                 hazmat_weights='models/hazmat.pt',
                 rust_model_path='models/deeplabv3_corrosion_multiclass.pth'):
        super().__init__()
        
        self.crack_weights = crack_weights
        self.hazmat_weights = hazmat_weights
        self.rust_model_path = rust_model_path
        
        # Detection state
        self.current_mode = self.MODES['STANDBY']
        self.running = False
        self.processing_thread = None
        
        # Frame buffer for processing
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Detectors (lazy initialization)
        self.crack_detector = None
        self.hazmat_detector = None
        self.qr_detector = None
        self.rust_detector = None
        self.motion_detector = None
        self.landolt_detector = None
        
        # GPU optimization
        self.gpu_manager = gpu_manager
        
        print(f"[AI] Detection System initialized with GPU: {self.gpu_manager.cuda_available}")
    
    def start(self):
        """Start AI detection processing"""
        if not self.running:
            self.running = True
            self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
            self.processing_thread.start()
            print("[AI] Detection system started")
    
    def stop(self):
        """Stop AI detection processing"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        print("[AI] Detection system stopped")
    
    def update_frame(self, frame: np.ndarray):
        """
        Update current frame for processing
        Called from video stream
        """
        with self.frame_lock:
            self.current_frame = frame.copy()
    
    def set_mode(self, mode: str):
        """Set detection mode"""
        if mode in self.MODES.values():
            self.current_mode = mode
            self.mode_changed.emit(mode)
            print(f"[AI] Mode changed to: {mode}")
    
    def _process_loop(self):
        """Main processing loop running in background thread"""
        while self.running:
            try:
                with self.frame_lock:
                    if self.current_frame is not None:
                        frame = self.current_frame.copy()
                    else:
                        time.sleep(0.01)
                        continue
                
                # Process frame based on current mode
                processed_frame = self._process_frame(frame)
                
                # Emit processed frame to UI
                if processed_frame is not None:
                    self.detection_frame_ready.emit(processed_frame)
                
                # Small delay to prevent overload
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                print(f"[AI] Processing error: {e}")
                time.sleep(0.1)
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame based on current mode
        Returns processed frame with annotations
        """
        if self.current_mode == self.MODES['STANDBY']:
            return self._draw_standby_overlay(frame)
        
        elif self.current_mode == self.MODES['CRACK']:
            return self._process_crack(frame)
        
        elif self.current_mode == self.MODES['HAZMAT']:
            return self._process_hazmat(frame)
        
        elif self.current_mode == self.MODES['QR']:
            return self._process_qr(frame)
        
        elif self.current_mode == self.MODES['LANDOLT']:
            return self._process_landolt(frame)
        
        elif self.current_mode == self.MODES['MOTION']:
            return self._process_motion(frame)
        
        elif self.current_mode == self.MODES['RUST']:
            return self._process_rust(frame)
        
        return frame
    
    def _draw_standby_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw standby mode overlay"""
        overlay = frame.copy()
        gpu_status = "GPU ON" if self.gpu_manager.cuda_available else "GPU OFF"
        cv2.putText(overlay, f"AI DETECTION - STANDBY [{gpu_status}]", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return overlay
    
    def _process_crack(self, frame: np.ndarray) -> np.ndarray:
        """Process crack detection"""
        # Initialize detector if needed
        if self.crack_detector is None:
            # self.crack_detector = CrackDetector(...)
            pass
        
        # Process with detector
        # processed_frame, _ = self.crack_detector.process_frame(frame)
        # return processed_frame
        
        # Placeholder
        overlay = frame.copy()
        cv2.putText(overlay, "CRACK DETECTION MODE", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return overlay
    
    def _process_hazmat(self, frame: np.ndarray) -> np.ndarray:
        """Process hazmat detection"""
        overlay = frame.copy()
        cv2.putText(overlay, "HAZMAT DETECTION MODE", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return overlay
    
    def _process_qr(self, frame: np.ndarray) -> np.ndarray:
        """Process QR detection"""
        overlay = frame.copy()
        cv2.putText(overlay, "QR DETECTION MODE", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        return overlay
    
    def _process_landolt(self, frame: np.ndarray) -> np.ndarray:
        """Process Landolt ring detection"""
        overlay = frame.copy()
        cv2.putText(overlay, "LANDOLT DETECTION MODE", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return overlay
    
    def _process_motion(self, frame: np.ndarray) -> np.ndarray:
        """Process motion detection"""
        overlay = frame.copy()
        cv2.putText(overlay, "MOTION DETECTION MODE", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return overlay
    
    def _process_rust(self, frame: np.ndarray) -> np.ndarray:
        """Process rust detection"""
        overlay = frame.copy()
        cv2.putText(overlay, "RUST DETECTION MODE", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)
        return overlay
    
    @staticmethod
    def numpy_to_qpixmap(frame: np.ndarray) -> QPixmap:
        """Convert numpy array to QPixmap for QLabel display"""
        if frame is None:
            return QPixmap()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # Create QImage
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to QPixmap
        return QPixmap.fromImage(q_image)