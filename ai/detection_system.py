"""
AI Detection System - Main controller with multi-camera support
"""

import cv2
import numpy as np
import time
import threading
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from .gpu_manager import gpu_manager


class AIDetectionSystem(QObject):
    """AI Detection System with 3-camera support"""
    
    # Signals for 3 cameras
    main_camera_frame_ready = pyqtSignal(np.ndarray)   # Main RTSP camera
    camera1_frame_ready = pyqtSignal(np.ndarray)       # Camera 1 (base1)
    camera2_frame_ready = pyqtSignal(np.ndarray)       # Camera 2 (base2)
    detection_status = pyqtSignal(str)
    mode_changed = pyqtSignal(str)
    
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
        
        # 3 camera frames
        self.camera_frames = {
            'main': None,      # Main RTSP
            'camera1': None,   # base1
            'camera2': None    # base2
        }
        self.frame_locks = {
            'main': threading.Lock(),
            'camera1': threading.Lock(),
            'camera2': threading.Lock()
        }
        
        # Processing threads
        self.processing_threads = {}
        
        # Detectors
        self.crack_detector = None
        self.hazmat_detector = None
        self.qr_detector = None
        self.rust_detector = None
        self.motion_detector = None
        self.landolt_detector = None
        
        # GPU
        self.gpu_manager = gpu_manager
        
        print(f"[AI] 3-Camera Detection System initialized with GPU: {self.gpu_manager.cuda_available}")
    
    def start(self):
        """Start AI detection for all 3 cameras"""
        if not self.running:
            self.running = True
            
            # Start thread for each camera
            for camera_id in ['main', 'camera1', 'camera2']:
                thread = threading.Thread(
                    target=self._process_loop, 
                    args=(camera_id,), 
                    daemon=True
                )
                thread.start()
                self.processing_threads[camera_id] = thread
            
            print("[AI] 3-Camera detection started")
    
    def stop(self):
        """Stop AI detection"""
        self.running = False
        for thread in self.processing_threads.values():
            if thread.is_alive():
                thread.join(timeout=2.0)
        self.processing_threads.clear()
        print("[AI] 3-Camera detection stopped")
    
    def update_frame(self, camera_id: str, frame: np.ndarray):
        """
        Update frame for specific camera
        camera_id: 'main', 'camera1', or 'camera2'
        """
        if camera_id in self.frame_locks:
            with self.frame_locks[camera_id]:
                self.camera_frames[camera_id] = frame.copy()
    
    def set_mode(self, mode: str):
        """Set detection mode (applies to all cameras)"""
        if mode in self.MODES.values():
            self.current_mode = mode
            self.mode_changed.emit(mode)
            print(f"[AI] Mode changed to: {mode} (all cameras)")
    
    def _process_loop(self, camera_id: str):
        """Processing loop for specific camera"""
        # Map camera to signal
        signal_map = {
            'main': self.main_camera_frame_ready,
            'camera1': self.camera1_frame_ready,
            'camera2': self.camera2_frame_ready
        }
        
        frame_signal = signal_map.get(camera_id)
        
        # Camera label untuk overlay
        camera_labels = {
            'main': 'MAIN CAM',
            'camera1': 'CAM 1',
            'camera2': 'CAM 2'
        }
        
        while self.running:
            try:
                with self.frame_locks[camera_id]:
                    if self.camera_frames[camera_id] is not None:
                        frame = self.camera_frames[camera_id].copy()
                    else:
                        time.sleep(0.01)
                        continue
                
                # Process frame
                processed_frame = self._process_frame(frame, camera_labels[camera_id])
                
                # Emit processed frame
                if processed_frame is not None and frame_signal:
                    frame_signal.emit(processed_frame)
                
                # ~30 FPS
                time.sleep(0.03)
                
            except Exception as e:
                print(f"[AI] {camera_id} error: {e}")
                time.sleep(0.1)
    
    def _process_frame(self, frame: np.ndarray, camera_label: str) -> np.ndarray:
        """Process frame based on current mode"""
        # Add camera label
        labeled_frame = frame.copy()
        cv2.putText(labeled_frame, camera_label, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        if self.current_mode == self.MODES['STANDBY']:
            return self._draw_standby_overlay(labeled_frame)
        elif self.current_mode == self.MODES['CRACK']:
            return self._process_crack(labeled_frame)
        elif self.current_mode == self.MODES['HAZMAT']:
            return self._process_hazmat(labeled_frame)
        elif self.current_mode == self.MODES['QR']:
            return self._process_qr(labeled_frame)
        elif self.current_mode == self.MODES['LANDOLT']:
            return self._process_landolt(labeled_frame)
        elif self.current_mode == self.MODES['MOTION']:
            return self._process_motion(labeled_frame)
        elif self.current_mode == self.MODES['RUST']:
            return self._process_rust(labeled_frame)
        
        return labeled_frame
    
    def _draw_standby_overlay(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        gpu_status = "GPU ON" if self.gpu_manager.cuda_available else "GPU OFF"
        cv2.putText(overlay, f"AI STANDBY [{gpu_status}]", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return overlay
    
    # Placeholder detection methods
    def _process_crack(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        cv2.putText(overlay, "CRACK DETECTION", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return overlay
    
    def _process_hazmat(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        cv2.putText(overlay, "HAZMAT DETECTION", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return overlay
    
    def _process_qr(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        cv2.putText(overlay, "QR DETECTION", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        return overlay
    
    def _process_landolt(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        cv2.putText(overlay, "LANDOLT DETECTION", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return overlay
    
    def _process_motion(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        cv2.putText(overlay, "MOTION DETECTION", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        return overlay
    
    def _process_rust(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        cv2.putText(overlay, "RUST DETECTION", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
        return overlay
    
    @staticmethod
    def numpy_to_qpixmap(frame: np.ndarray) -> QPixmap:
        if frame is None:
            return QPixmap()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)