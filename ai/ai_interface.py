"""
AI Detection Interface
Simple interface untuk integrasi dengan main window
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
import os

from .controller import AIDetectionController


class AIDetector:
    """
    Real-time AI detector wrapper
    Untuk objek imgDetector di main window
    """
    
    def __init__(self, controller: AIDetectionController):
        self.controller = controller
        self.last_mode = None
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process frame untuk real-time detection
        
        Args:
            frame: Input frame (BGR format dari OpenCV)
            
        Returns:
            Tuple[processed_frame, analysis_frame]
            - processed_frame: Frame dengan annotations
            - analysis_frame: Analysis view (hanya untuk Rust, None untuk lainnya)
        """
        return self.controller.process_frame(frame)
    
    def get_current_mode(self) -> str:
        """Get current detection mode"""
        return self.controller.current_mode
    
    def get_detection_stats(self) -> Dict[str, int]:
        """Get detection statistics"""
        stats = {}
        
        if self.controller.crack_detector:
            stats['crack'] = self.controller.crack_detector.get_capture_count()
        
        if self.controller.hazmat_detector:
            stats['hazmat'] = len(self.controller.hazmat_detector.saved_objects)
        
        if self.controller.qr_detector:
            stats['qr'] = len(self.controller.qr_detector.saved_qr_codes)
        
        if self.controller.landolt_detector:
            stats['landolt'] = self.controller.landolt_detector.get_capture_count()
        
        if self.controller.motion_detector:
            stats['motion'] = self.controller.motion_detector.get_capture_count()
        
        if self.controller.rust_detector:
            stats['rust'] = self.controller.rust_detector.get_capture_count()
        
        return stats
    
    def switch_mode(self, mode: str) -> bool:
        """
        Manually switch detection mode
        
        Args:
            mode: Mode name ('crack', 'hazmat', 'qr', 'landolt', 'motion', 'rust', 'standby')
            
        Returns:
            True if mode switched successfully
        """
        mode_map = {
            'crack': self.controller.MODES['CRACK'],
            'hazmat': self.controller.MODES['HAZMAT'],
            'qr': self.controller.MODES['QR'],
            'landolt': self.controller.MODES['LANDOLT'],
            'motion': self.controller.MODES['MOTION'],
            'rust': self.controller.MODES['RUST'],
            'standby': self.controller.MODES['STANDBY']
        }
        
        if mode.lower() in mode_map:
            self.controller.switch_mode(mode_map[mode.lower()])
            return True
        return False


class AICapturer:
    """
    AI capture handler
    Untuk objek imgCapture di main window
    """
    
    def __init__(self, controller: AIDetectionController):
        self.controller = controller
    
    def save_current_detection(self) -> Dict[str, Any]:
        """
        Save current detection result
        
        Returns:
            Dict dengan info save result:
            - success: bool
            - mode: detection mode
            - file_path: path to saved file
            - message: info message
        """
        current_mode = self.controller.current_mode
        
        result = {
            'success': False,
            'mode': current_mode,
            'file_path': None,
            'message': 'No detection active'
        }
        
        try:
            # Untuk setiap mode, trigger save jika ada detection active
            if current_mode == self.controller.MODES['CRACK'] and self.controller.crack_detector:
                # Crack sudah auto-save, return info saja
                result['success'] = True
                result['message'] = 'Crack detection auto-saved'
                result['file_path'] = 'ai_crack_captures/'
                
            elif current_mode == self.controller.MODES['HAZMAT'] and self.controller.hazmat_detector:
                result['success'] = True
                result['message'] = 'Hazmat detection auto-saved'
                result['file_path'] = 'ai_hazmat_images/'
                
            elif current_mode == self.controller.MODES['QR'] and self.controller.qr_detector:
                result['success'] = True
                result['message'] = 'QR detection auto-saved'
                result['file_path'] = 'ai_qr_captures/'
                
            elif current_mode == self.controller.MODES['LANDOLT'] and self.controller.landolt_detector:
                result['success'] = True
                result['message'] = 'Landolt Ring detection auto-saved'
                result['file_path'] = 'ai_landolt_captures/'
                
            elif current_mode == self.controller.MODES['MOTION'] and self.controller.motion_detector:
                result['success'] = True
                result['message'] = 'Motion detection auto-saved'
                result['file_path'] = 'ai_motion_captures/'
                
            elif current_mode == self.controller.MODES['RUST'] and self.controller.rust_detector:
                result['success'] = True
                result['message'] = 'Rust detection auto-saved with analysis'
                result['file_path'] = 'ai_rust_captures/'
            
        except Exception as e:
            result['message'] = f'Error saving: {e}'
        
        return result
    
    def get_capture_count(self, mode: Optional[str] = None) -> int:
        """
        Get capture count untuk mode tertentu atau current mode
        
        Args:
            mode: Mode name (optional, default current mode)
            
        Returns:
            Capture count
        """
        if mode is None:
            mode = self.controller.current_mode
        
        mode_map = {
            'crack': lambda: self.controller.crack_detector.get_capture_count() if self.controller.crack_detector else 0,
            'hazmat': lambda: len(self.controller.hazmat_detector.saved_objects) if self.controller.hazmat_detector else 0,
            'qr': lambda: len(self.controller.qr_detector.saved_qr_codes) if self.controller.qr_detector else 0,
            'landolt': lambda: self.controller.landolt_detector.get_capture_count() if self.controller.landolt_detector else 0,
            'motion': lambda: self.controller.motion_detector.get_capture_count() if self.controller.motion_detector else 0,
            'rust': lambda: self.controller.rust_detector.get_capture_count() if self.controller.rust_detector else 0,
        }
        
        for key, func in mode_map.items():
            if key in mode.lower():
                return func()
        
        return 0
    
    def clear_all_captures(self) -> Dict[str, bool]:
        """
        Clear all capture history
        
        Returns:
            Dict dengan status clear untuk setiap mode
        """
        result = {}
        
        try:
            if self.controller.crack_detector:
                self.controller.crack_detector.capture_count = 0
                result['crack'] = True
            
            if self.controller.hazmat_detector:
                self.controller.hazmat_detector.saved_objects.clear()
                self.controller.hazmat_detector.detection_count.clear()
                result['hazmat'] = True
            
            if self.controller.qr_detector:
                self.controller.qr_detector.saved_qr_codes.clear()
                self.controller.qr_detector.detection_count.clear()
                result['qr'] = True
            
            if self.controller.landolt_detector:
                self.controller.landolt_detector.save_counter = 0
                self.controller.landolt_detector.saved_positions.clear()
                result['landolt'] = True
            
            if self.controller.motion_detector:
                self.controller.motion_detector.save_counter = 0
                result['motion'] = True
            
            if self.controller.rust_detector:
                self.controller.rust_detector.capture_count = 0
                result['rust'] = True
            
        except Exception as e:
            print(f"Error clearing captures: {e}")
        
        return result


class AIDetectionSystem:
    """
    Main AI Detection System
    Interface utama untuk main window
    """
    
    def __init__(
        self,
        crack_weights: str = "models/crack.pt",
        hazmat_weights: str = "models/hazmat.pt",
        rust_model_path: str = "models/deeplabv3_corrosion_multiclass.pth",
        camera_index: int = 0
    ):
        """
        Initialize AI Detection System
        
        Args:
            crack_weights: Path to crack detection model
            hazmat_weights: Path to hazmat detection model
            rust_model_path: Path to rust detection model
            camera_index: Camera index (not used in this interface)
        """
        print("Initializing AI Detection System...")
        
        # Verify model files
        self._verify_models(crack_weights, hazmat_weights, rust_model_path)
        
        # Initialize controller tanpa camera (akan terima frame dari external)
        self.controller = AIDetectionController(
            crack_weights=crack_weights,
            hazmat_weights=hazmat_weights,
            rust_model_path=rust_model_path,
            camera_index=camera_index,
            external_feed=True  # Flag bahwa camera external
        )
        
        # Initialize interfaces
        self.detector = AIDetector(self.controller)
        self.capturer = AICapturer(self.controller)
        
        print("✓ AI Detection System initialized")
        print(f"  GPU: {'ENABLED' if self.controller.gpu_manager.cuda_available else 'DISABLED'}")
    
    def _verify_models(self, crack_weights, hazmat_weights, rust_model_path):
        """Verify model files exist"""
        missing = []
        
        if not os.path.exists(crack_weights):
            missing.append(f"Crack model: {crack_weights}")
        
        if not os.path.exists(hazmat_weights):
            missing.append(f"Hazmat model: {hazmat_weights}")
        
        if not os.path.exists(rust_model_path):
            missing.append(f"Rust model: {rust_model_path}")
        
        if missing:
            print("⚠ Missing models:")
            for m in missing:
                print(f"  - {m}")
            print("System will run with available models only.")
    
    def start(self):
        """Start AI detection services"""
        return self.controller.start_services()
    
    def stop(self):
        """Stop AI detection services"""
        self.controller.cleanup()
    
    def get_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'gpu_available': self.controller.gpu_manager.cuda_available,
            'current_mode': self.controller.current_mode,
            'detection_stats': self.detector.get_detection_stats(),
            'frame_count': self.controller.frame_count,
            'fps': self.controller.fps
        }