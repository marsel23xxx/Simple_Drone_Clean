import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models.segmentation as models
from PIL import Image
import datetime
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..gpu_manager import gpu_manager


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
class PositionData:
    x_distance_m: float
    y_distance_m: float
    z_height_ground_m: float
    square_size_m: float
    angle_horizontal: float
    angle_vertical: float


class RustDetector:
    def __init__(self, model_path='models/deeplabv3_corrosion_multiclass.pth', 
                 save_folder="ai_rust_captures", camera_height_m=1.5):
        self.WARP_SIZE = 300
        self.FRAME_SIZE = (640, 480)
        self.MIN_AREA = 5000
        self.NUM_CLASSES = 4
        self.save_folder = save_folder
        self.camera_height_m = camera_height_m
        
        os.makedirs(self.save_folder, exist_ok=True)
        
        self.device = gpu_manager.device
        self.class_colors = {1: (0, 255, 0), 2: (0, 255, 255), 3: (0, 0, 255)}
        
        self.transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
        
        self.model = models.deeplabv3_resnet50(weights=None, num_classes=self.NUM_CLASSES)
        
        if gpu_manager.cuda_available:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint)
        
        self.model = gpu_manager.optimize_model(self.model)
        self.model.eval()
        
        self.results_data: List[CorrosionData] = []
        self.position_data = None
        self.capture_count = 0
        
        print(f"Rust Detector initialized - Output: {self.save_folder}")
    
    def process_frame(self, frame: np.ndarray) -> dict:
        square_points = self._detect_square(frame)
        
        if square_points is not None:
            rect = self._order_points(square_points.reshape(4, 2))
            M = cv2.getPerspectiveTransform(
                np.float32(rect),
                np.float32([[0, 0], [self.WARP_SIZE, 0], 
                           [self.WARP_SIZE, self.WARP_SIZE], [0, self.WARP_SIZE]])
            )
            
            warped_square = cv2.warpPerspective(frame, M, (self.WARP_SIZE, self.WARP_SIZE))
            
            rust_img, clean_img = self._process_corrosion(warped_square)
            analysis_frame = self._create_display(rust_img, clean_img)
            
            main_frame = frame.copy()
            cv2.drawContours(main_frame, [square_points], -1, (0, 255, 0), 2)
            
            return {
                'main_frame': main_frame,
                'analysis_frame': analysis_frame,
                'has_analysis': True
            }
        
        return {
            'main_frame': frame.copy(),
            'analysis_frame': None,
            'has_analysis': False
        }
    
    def _detect_square(self, frame: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > self.MIN_AREA:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4:
                    return approx
        return None
    
    def _process_corrosion(self, warped: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.results_data.clear()
        overlayed = warped.copy()
        clean = warped.copy()
        
        image_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(image_pil).unsqueeze(0)
        
        if gpu_manager.cuda_available:
            input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out']
            mask = torch.argmax(output.squeeze(), dim=0)
            if mask.is_cuda:
                mask = mask.cpu().numpy().astype(np.uint8)
            else:
                mask = mask.numpy().astype(np.uint8)
        
        del input_tensor, output
        if gpu_manager.cuda_available:
            torch.cuda.empty_cache()
        
        mask_resized = cv2.resize(mask, (self.WARP_SIZE, self.WARP_SIZE), interpolation=cv2.INTER_NEAREST)
        
        for class_id, color in self.class_colors.items():
            overlayed[mask_resized == class_id] = color
        
        self.results_data.extend(self._extract_measurements(mask_resized))
        
        return overlayed, clean
    
    def _extract_measurements(self, mask: np.ndarray) -> List[CorrosionData]:
        total_pixels = mask.shape[0] * mask.shape[1]
        severity_percentages = {}
        total_affected = 0
        
        unique, counts = np.unique(mask, return_counts=True)
        pixel_counts = dict(zip(unique, counts))
        
        for class_id in range(1, self.NUM_CLASSES):
            count = pixel_counts.get(class_id, 0)
            percentage = (count / total_pixels) * 100
            if percentage > 0.1:
                severity_percentages[class_id] = percentage
                total_affected += count
        
        if not severity_percentages:
            return []
        
        total_affected_percentage = (total_affected / total_pixels) * 100
        return [CorrosionData(severity_percentages, total_affected_percentage)]
    
    def _create_display(self, rust_img: np.ndarray, clean_img: np.ndarray) -> np.ndarray:
        h, w = rust_img.shape[:2]
        margin, gap = 50, 50
        combined = np.ones((h + 2 * margin, margin + w + gap + w + margin, 3), dtype=np.uint8) * 255
        
        combined[margin:margin + h, margin:margin + w] = rust_img
        right_start = margin + w + gap
        combined[margin:margin + h, right_start:right_start + w] = clean_img
        
        if self.results_data:
            corrosion = self.results_data[0]
            info_text = f"Rust: {corrosion.total_affected_percentage:.1f}% | {corrosion.dominant_class}"
            cv2.putText(combined, info_text, (margin, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return combined
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d = pts[:, 0] - pts[:, 1]
        rect[1] = pts[np.argmax(d)]
        rect[3] = pts[np.argmin(d)]
        return rect
    
    def get_capture_count(self) -> int:
        return self.capture_count

