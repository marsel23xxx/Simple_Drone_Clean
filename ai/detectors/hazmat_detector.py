"""
Hazmat Detection Module
GPU-optimized hazmat detection using YOLO
"""

import cv2
import numpy as np
import math
import time
import datetime
import os
from ultralytics import YOLO
from ..gpu_manager import gpu_manager


class HazmatDetector:
    def __init__(self, model_path="models/hazmat.pt"):
        # GPU-optimized YOLO model loading
        self.device = gpu_manager.device
        self.model = YOLO(model_path)
        
        # Frame dimensions
        self.frame_width = 640
        self.frame_height = 480
        self.frame_center_x = self.frame_width // 2
        self.frame_center_y = self.frame_height // 2
        
        # Camera parameters
        self.CAMERA_FOV_HORIZONTAL = 60
        self.CAMERA_FOV_VERTICAL = 45
        self.CAMERA_HEIGHT = 1.5
        self.PIXELS_PER_M_AT_1M = 500
        
        self.class_names = getattr(self.model, 'names', {})
        
        # Hazard classifications
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
        
        self.CONFIDENCE_THRESHOLD = 0.5
        self.STABILIZATION_FRAMES = 3
        self.detection_count = {}
        self.saved_objects = set()
        
        # Create folders
        os.makedirs("ai_hazmat_images", exist_ok=True)
        os.makedirs("ai_hazmat_analytics", exist_ok=True)
        
        print(f"[Hazmat] Detector initialized with GPU: {gpu_manager.cuda_available}")
    
    def update_frame_dimensions(self, width, height):
        """Update frame dimensions for coordinate calculations"""
        self.frame_width = width
        self.frame_height = height
        self.frame_center_x = width // 2
        self.frame_center_y = height // 2
    
    def calculate_real_world_coordinates(self, box):
        """Calculate real world position in meters"""
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
        height_z = self.CAMERA_HEIGHT + height_offset
        height_z = max(0.0, min(3.0, height_z))
        
        if horizontal_angle < -0.5:
            direction = "Left"
        elif horizontal_angle > 0.5:
            direction = "Right"
        else:
            direction = "Center"
        
        return (round(distance_x, 2), round(horizontal_y, 2), round(height_z, 2), 
                round(horizontal_angle, 1), round(vertical_angle, 1), direction)
    
    def create_object_id(self, box, class_name):
        """Create unique object ID"""
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return f"{class_name}_{center_x//50}_{center_y//50}"
    
    def save_detection(self, frame, detections_data):
        """Save hazmat detection with analytics"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        image_filename = f"ai_hazmat_images/hazmat_{timestamp}.jpg"
        cv2.imwrite(image_filename, frame)
        
        analytics_filename = f"ai_hazmat_analytics/hazmat_{timestamp}.txt"
        with open(analytics_filename, 'w') as f:
            f.write(f"Hazmat Detection Analysis (GPU Optimized)\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Image File: {image_filename}\n")
            f.write(f"Frame Size: {self.frame_width}x{self.frame_height}\n")
            f.write(f"GPU Acceleration: {'ENABLED' if gpu_manager.cuda_available else 'DISABLED'}\n")
            f.write(f"Total Objects Detected: {len(detections_data)}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, detection in enumerate(detections_data, 1):
                f.write(f"Object {i}:\n")
                f.write(f"  Hazard Type: {detection['hazard_class']}\n")
                f.write(f"  Confidence: {detection['confidence']}\n")
                f.write(f"  Position Analysis:\n")
                f.write(f"    Direction: {detection['direction']}\n")
                f.write(f"    Distance X: {detection['distance_x']:.2f}m\n")
                f.write(f"    Position Y: {detection['position_y']:+.2f}m\n")
                f.write(f"    Height Z: {detection['height_z']:.2f}m\n")
                f.write(f"    Horizontal Angle: {detection['horizontal_angle']:+.1f}°\n")
                f.write(f"    Vertical Angle: {detection['vertical_angle']:+.1f}°\n")
                f.write("-" * 30 + "\n\n")
        
        print(f"[Hazmat] Saved: {image_filename}")
        return analytics_filename
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame for hazmat detection"""
        import torch
        
        # Update dimensions
        self.update_frame_dimensions(frame.shape[1], frame.shape[0])
        
        # Run detection with GPU
        with torch.no_grad():
            results = self.model(frame, device=self.device, verbose=False)[0]
        
        # Annotate frame
        annotated = self.annotate_frame(frame, results)
        
        return annotated
    
    def annotate_frame(self, frame, results):
        """Annotate frame with detections"""
        annotated = frame.copy()
        detections_to_save = []
        current_objects = set()
        frame_height, frame_width = frame.shape[:2]
        
        if results.boxes is None or len(results.boxes) == 0:
            return annotated
        
        detected_objects = []
        
        for box in results.boxes:
            confidence = float(box.conf[0])
            if confidence < self.CONFIDENCE_THRESHOLD:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            class_name = self.class_names.get(cls_id, str(cls_id))
            hazard_label = self.hazard_classes.get(class_name, class_name)
            
            if len(hazard_label) > 25:
                hazard_label = hazard_label[:22] + "..."
            
            distance_x, position_y, height_z, horizontal_angle, vertical_angle, direction = \
                self.calculate_real_world_coordinates(box)
            
            object_id = self.create_object_id(box, class_name)
            current_objects.add(object_id)
            
            if object_id not in self.detection_count:
                self.detection_count[object_id] = 0
            self.detection_count[object_id] += 1
            
            detected_objects.append({
                'box': (x1, y1, x2, y2),
                'label': hazard_label,
                'confidence': confidence,
                'object_id': object_id,
                'class_name': class_name,
                'spatial_data': (distance_x, position_y, height_z, horizontal_angle, vertical_angle, direction)
            })
        
        # Sort by y-coordinate
        detected_objects.sort(key=lambda obj: obj['box'][1])
        
        used_positions = []
        
        for obj in detected_objects:
            x1, y1, x2, y2 = obj['box']
            hazard_label = obj['label']
            confidence = obj['confidence']
            object_id = obj['object_id']
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Position text
            text_y = y1 - 10
            text_x = x1
            
            text_height = 25
            text_width = len(hazard_label) * 12
            
            # Check overlap
            for used_pos in used_positions:
                used_x, used_y, used_w, used_h = used_pos
                if (text_x < used_x + used_w and text_x + text_width > used_x and 
                    text_y < used_y + used_h and text_y + text_height > used_y):
                    text_y = y2 + 25
                    break
            
            if text_y < 25:
                text_y = y2 + 25
            if text_y > frame_height - 10:
                text_y = y1 - 10
            
            text_x = max(5, min(text_x, frame_width - text_width - 10))
            
            # Background
            bg_padding = 5
            bg_x1 = max(0, text_x - bg_padding)
            bg_y1 = max(0, text_y - 20)
            bg_x2 = min(frame_width, text_x + text_width + bg_padding)
            bg_y2 = min(frame_height, text_y + 10)
            
            overlay = annotated.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
            
            # Text
            cv2.putText(annotated, hazard_label, 
                       (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            conf_text = f"Conf: {confidence:.2f}"
            cv2.putText(annotated, conf_text, 
                       (text_x, text_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            used_positions.append((text_x, text_y - 20, text_width, text_height))
            
            # Save logic
            distance_x, position_y, height_z, horizontal_angle, vertical_angle, direction = obj['spatial_data']
            
            if (self.detection_count[object_id] >= self.STABILIZATION_FRAMES and 
                object_id not in self.saved_objects):
                
                self.saved_objects.add(object_id)
                
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
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "width": x2-x1, "height": y2-y1,
                        "center_x": (x1+x2)//2, "center_y": (y1+y2)//2
                    }
                }
                detections_to_save.append(detection_data)
                
                save_indicator_y = min(y2 + 40, frame_height - 5)
                cv2.putText(annotated, "SAVED", (text_x, save_indicator_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Cleanup
        objects_to_remove = set(self.detection_count.keys()) - current_objects
        for obj_id in objects_to_remove:
            del self.detection_count[obj_id]
            self.saved_objects.discard(obj_id)
        
        # Save
        if detections_to_save:
            self.save_detection(annotated, detections_to_save)
        
        return annotated