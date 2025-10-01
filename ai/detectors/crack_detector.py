"""
Crack Detection Module
GPU-optimized crack detection using YOLO
"""

import cv2
import numpy as np
import math
import time
import datetime
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
from ultralytics import YOLO

from ..gpu_manager import gpu_manager


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


class CrackDetector:
    def __init__(self, weights='models/crack.pt', conf_thresh=0.80, output_dir='ai_crack_captures'):
        self.WARP_SIZE = 300
        self.M_PER_PIXEL = 0.10 / self.WARP_SIZE
        self.MIN_AREA = 5000
        
        # Camera parameters
        self.CAMERA_HEIGHT = 1.500
        self.CAMERA_FOV_H = 60.0
        self.CAMERA_FOV_V = 45.0
        self.DETECTED_SQUARE_SIZE = 0.100
        
        # GPU-optimized model loading
        self.device = gpu_manager.device
        self.model = YOLO(weights)
        self.model.fuse()
        self.model.overrides['conf'] = conf_thresh
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.captured_square = None
        self.captured_original = None
        self.results_data: List[CrackData] = []
        self.current_display = None
        self.capture_count = 0
        self.current_timestamp = ""
        self.current_frame_size = (0, 0)
        
        # Frame buffer
        self.frame_buffer = []
        self.buffer_size = 10
        self.blur_threshold = 100
        self.detection_frames = 0
        self.last_capture_time = 0
        self.capture_cooldown = 0.1
        
        self.crack_detection_active = False
        self.processing_start_time = 0
        self.crack_check_frames = 0
        
        print(f"[Crack] Detector initialized with GPU: {gpu_manager.cuda_available}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Process frame for crack detection"""
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        self.current_frame_size = (frame.shape[1], frame.shape[0])
        
        square_points = self._detect_square(frame)
        
        if square_points is not None:
            self.detection_frames += 1
            is_stable = self._is_frame_stable(frame, square_points)
            
            status_frame = frame.copy()
            cv2.drawContours(status_frame, [square_points], -1, (0, 255, 0), 2)
            
            time_since_capture = current_time - self.last_capture_time
            can_capture = time_since_capture >= self.capture_cooldown
            
            if is_stable and can_capture:
                if not self.crack_detection_active:
                    self.crack_detection_active = True
                    self.processing_start_time = current_time
                    self.crack_check_frames = 0
                    status_text = "ANALYZING..."
                    status_color = (0, 255, 255)
                else:
                    self.crack_check_frames += 1
                    elapsed_analysis = current_time - self.processing_start_time
                    
                    status_text = f"ANALYSIS {elapsed_analysis:.1f}s"
                    status_color = (0, 255, 255)
                    
                    if self.crack_check_frames >= 5:
                        best_frame_info = self._get_best_frame_from_buffer()
                        if best_frame_info:
                            best_frame = best_frame_info['frame']
                            best_square_points = best_frame_info['square_points']
                            
                            rect = self._order_points(best_square_points.reshape(4, 2))
                            M = cv2.getPerspectiveTransform(
                                np.float32(rect),
                                np.float32([[0, 0], [self.WARP_SIZE, 0], 
                                           [self.WARP_SIZE, self.WARP_SIZE], [0, self.WARP_SIZE]])
                            )
                            
                            warped_square = cv2.warpPerspective(best_frame, M, (self.WARP_SIZE, self.WARP_SIZE))
                            cracks_detected = self._check_for_cracks(warped_square)
                            
                            if cracks_detected:
                                self.captured_square = warped_square
                                self.captured_original = best_frame.copy()
                                
                                crack_img, clean_img = self._process_cracks()
                                self.current_display = self._create_display(crack_img, clean_img)
                                
                                self._auto_save_capture(best_frame, best_square_points)
                                
                                self.last_capture_time = current_time
                                self.capture_count += 1
                                
                                self.frame_buffer.clear()
                                self.detection_frames = 0
                                self.crack_detection_active = False
                                
                                success_frame = best_frame.copy()
                                cv2.drawContours(success_frame, [best_square_points], -1, (0, 255, 0), 3)
                                cv2.putText(success_frame, "CRACK DETECTED", 
                                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.putText(success_frame, f"SAVED #{self.capture_count}", 
                                           (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                                
                                return success_frame, self.current_display
                            else:
                                self.crack_detection_active = False
                                self.frame_buffer.clear()
                                self.detection_frames = 0
                                
                                no_crack_frame = best_frame.copy()
                                cv2.drawContours(no_crack_frame, [best_square_points], -1, (0, 165, 255), 2)
                                cv2.putText(no_crack_frame, "NO CRACKS", 
                                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                                
                                return no_crack_frame, None
                                
            elif not can_capture:
                remaining_time = self.capture_cooldown - time_since_capture
                status_text = f"COOLDOWN {remaining_time:.1f}s"
                status_color = (0, 255, 255)
                self.crack_detection_active = False
            else:
                self.crack_detection_active = False
                sharpness = self._calculate_frame_sharpness(frame)
                if sharpness < self.blur_threshold:
                    status_text = "STABILIZING..."
                    status_color = (0, 165, 255)
                else:
                    status_text = "HOLD STEADY"
                    status_color = (0, 255, 255)
            
            cv2.putText(status_frame, status_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            
            return status_frame, None
        else:
            self.detection_frames = 0
            self.frame_buffer.clear()
            self.crack_detection_active = False
            
            live_frame = frame.copy()
            cv2.putText(live_frame, "SCANNING FOR CRACKS", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            return live_frame, None
    
    def _check_for_cracks(self, warped_square: np.ndarray) -> bool:
        """GPU-optimized crack detection"""
        import torch
        with torch.no_grad():
            results = self.model.predict(warped_square, imgsz=640, verbose=False, device=self.device)
        
        crack_found = False
        for r in results:
            if r.masks is not None:
                if hasattr(r.masks.data, 'cpu'):
                    masks = r.masks.data.cpu().numpy()
                else:
                    masks = r.masks.data.numpy() if hasattr(r.masks.data, 'numpy') else r.masks.data
                
                for mask in masks:
                    mask_resized = cv2.resize((mask * 255).astype(np.uint8), 
                                            (self.WARP_SIZE, self.WARP_SIZE))
                    
                    crack_pixels = np.sum(mask_resized > 127)
                    
                    if crack_pixels > 50:
                        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for cnt in contours:
                            if cv2.contourArea(cnt) > 30:
                                crack_found = True
                                break
                        
                        if crack_found:
                            break
                
                if crack_found:
                    break
        
        return crack_found
    
    def _calculate_frame_sharpness(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    
    def _is_frame_stable(self, frame: np.ndarray, square_points: np.ndarray) -> bool:
        sharpness = self._calculate_frame_sharpness(frame)
        
        frame_info = {
            'frame': frame.copy(),
            'square_points': square_points.copy(),
            'sharpness': sharpness,
            'timestamp': cv2.getTickCount()
        }
        
        self.frame_buffer.append(frame_info)
        
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        
        if len(self.frame_buffer) < 5:
            return False
        
        if sharpness < self.blur_threshold:
            return False
        
        recent_frames = self.frame_buffer[-5:]
        positions = [info['square_points'].reshape(4, 2) for info in recent_frames]
        
        if len(positions) >= 2:
            pos_array = np.array(positions)
            variance = np.var(pos_array, axis=0)
            max_variance = np.max(variance)
            
            if max_variance > 50:
                return False
        
        return True
    
    def _get_best_frame_from_buffer(self) -> Optional[dict]:
        if not self.frame_buffer:
            return None
        
        good_frames = [f for f in self.frame_buffer if f['sharpness'] >= self.blur_threshold]
        
        if not good_frames:
            return max(self.frame_buffer, key=lambda x: x['sharpness'])
        
        return max(good_frames, key=lambda x: x['sharpness'])
    
    def _detect_square(self, frame: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > self.MIN_AREA:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4:
                    return approx
        return None
    
    def _process_cracks(self) -> Tuple[np.ndarray, np.ndarray]:
        import torch
        self.results_data.clear()
        overlayed = self.captured_square.copy()
        clean = self.captured_square.copy()
        
        with torch.no_grad():
            results = self.model.predict(self.captured_square, imgsz=640, verbose=False, device=self.device)
        
        for r in results:
            if r.masks is None:
                continue
                
            if hasattr(r.masks.data, 'cpu'):
                masks = r.masks.data.cpu().numpy()
            else:
                masks = r.masks.data.numpy() if hasattr(r.masks.data, 'numpy') else r.masks.data
                
            if r.boxes is not None:
                if hasattr(r.boxes.conf, 'cpu'):
                    confidences = r.boxes.conf.cpu().numpy()
                else:
                    confidences = r.boxes.conf.numpy() if hasattr(r.boxes.conf, 'numpy') else r.boxes.conf
            else:
                confidences = [0.8] * len(masks)
            
            for idx, mask in enumerate(masks):
                mask_resized = cv2.resize((mask * 255).astype(np.uint8), 
                                        (self.WARP_SIZE, self.WARP_SIZE))
                overlayed = self._apply_mask(overlayed, mask_resized)
                
                crack_measurements = self._extract_measurements(mask_resized)
                for crack in crack_measurements:
                    if idx < len(confidences):
                        crack.confidence = float(confidences[idx])
                    else:
                        crack.confidence = 0.8
                
                self.results_data.extend(crack_measurements)
        
        cv2.rectangle(overlayed, (0, 0), (self.WARP_SIZE-1, self.WARP_SIZE-1), (0, 255, 0), 3)
        cv2.rectangle(clean, (0, 0), (self.WARP_SIZE-1, self.WARP_SIZE-1), (0, 0, 0), 1)
        
        return overlayed, clean
    
    def _apply_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mask_bool = mask > 127
        if np.any(mask_bool):
            frame[mask_bool] = [0, 0, 255]
        return frame
    
    def _extract_measurements(self, mask: np.ndarray) -> List[CrackData]:
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
        
        endpoints = self._find_endpoints(cnt)
        
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
    
    def _find_endpoints(self, contour: np.ndarray) -> List[Tuple[int, int]]:
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
    
    def _create_display(self, crack_img: np.ndarray, clean_img: np.ndarray) -> np.ndarray:
        h, w = crack_img.shape[:2]
        margin = 50
        gap = 50
        total_width = margin + w + gap + w + margin
        total_height = h + 2 * margin
        combined = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
        
        combined[margin:margin + h, margin:margin + w] = crack_img
        right_start = margin + w + gap
        combined[margin:margin + h, right_start:right_start + w] = clean_img
        
        return combined
    
    def _auto_save_capture(self, frame: np.ndarray, square_points: np.ndarray):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_timestamp = timestamp
        capture_folder = os.path.join(self.output_dir, f"crack_capture_{timestamp}")
        os.makedirs(capture_folder, exist_ok=True)
        
        final_frame = frame.copy()
        cv2.drawContours(final_frame, [square_points], -1, (0, 255, 0), 3)
        
        cv2.putText(final_frame, f"CRACK CAPTURE #{self.capture_count}", 
                   (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(final_frame, timestamp, 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        paths = {
            'original': os.path.join(capture_folder, f"crack_original_{timestamp}.jpg"),
            'warped_display': os.path.join(capture_folder, f"crack_analysis_{timestamp}.jpg")
        }
        
        cv2.imwrite(paths['original'], final_frame)
        if self.current_display is not None:
            cv2.imwrite(paths['warped_display'], self.current_display)
        
        print(f"[Crack] Capture saved to: {capture_folder}")
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()
        
        rect[0] = pts[np.argmin(s)]
        rect[1] = pts[np.argmin(d)]
        rect[2] = pts[np.argmax(s)]
        rect[3] = pts[np.argmax(d)]
        
        return rect
    
    def get_capture_count(self) -> int:
        return self.capture_count