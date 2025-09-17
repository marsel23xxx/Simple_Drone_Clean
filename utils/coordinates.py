"""
Coordinate transformation and utility functions
Handles various coordinate system conversions and calculations
"""

import math
import numpy as np
from typing import Tuple, List, Optional


def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return math.radians(degrees)


def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees."""
    return math.degrees(radians)


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-π, π] range."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def calculate_distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate 2D Euclidean distance between two points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def calculate_distance_3d(p1: Tuple[float, float, float], 
                         p2: Tuple[float, float, float]) -> float:
    """Calculate 3D Euclidean distance between two points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def calculate_bearing(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate bearing (angle) from p1 to p2 in radians."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.atan2(dy, dx)


def rotate_point_2d(point: Tuple[float, float], 
                   center: Tuple[float, float], 
                   angle: float) -> Tuple[float, float]:
    """Rotate a 2D point around a center by given angle (radians)."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    # Translate to origin
    x = point[0] - center[0]
    y = point[1] - center[1]
    
    # Rotate
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a
    
    # Translate back
    return (x_rot + center[0], y_rot + center[1])


def world_to_grid(world_coords: Tuple[float, float], 
                 grid_size: float = 1.0,
                 origin: Tuple[float, float] = (0, 0)) -> Tuple[int, int]:
    """Convert world coordinates to grid coordinates."""
    grid_x = int((world_coords[0] - origin[0]) / grid_size)
    grid_y = int((world_coords[1] - origin[1]) / grid_size)
    return (grid_x, grid_y)


def grid_to_world(grid_coords: Tuple[int, int], 
                 grid_size: float = 1.0,
                 origin: Tuple[float, float] = (0, 0)) -> Tuple[float, float]:
    """Convert grid coordinates to world coordinates."""
    world_x = grid_coords[0] * grid_size + origin[0]
    world_y = grid_coords[1] * grid_size + origin[1]
    return (world_x, world_y)


def local_to_global(local_point: Tuple[float, float],
                   reference_point: Tuple[float, float],
                   reference_heading: float) -> Tuple[float, float]:
    """Convert local coordinates to global coordinates."""
    # Rotate local point by reference heading
    rotated = rotate_point_2d(local_point, (0, 0), reference_heading)
    
    # Translate to global position
    global_x = rotated[0] + reference_point[0]
    global_y = rotated[1] + reference_point[1]
    
    return (global_x, global_y)


def global_to_local(global_point: Tuple[float, float],
                   reference_point: Tuple[float, float],
                   reference_heading: float) -> Tuple[float, float]:
    """Convert global coordinates to local coordinates."""
    # Translate to local origin
    local_x = global_point[0] - reference_point[0]
    local_y = global_point[1] - reference_point[1]
    
    # Rotate by negative reference heading
    local_point = rotate_point_2d((local_x, local_y), (0, 0), -reference_heading)
    
    return local_point


def interpolate_waypoints(waypoints: List[Tuple[float, float]], 
                         resolution: float = 0.1) -> List[Tuple[float, float]]:
    """Interpolate between waypoints with given resolution."""
    if len(waypoints) < 2:
        return waypoints.copy()
    
    interpolated = [waypoints[0]]
    
    for i in range(1, len(waypoints)):
        start = waypoints[i-1]
        end = waypoints[i]
        
        distance = calculate_distance_2d(start, end)
        steps = int(distance / resolution)
        
        for step in range(1, steps + 1):
            t = step / steps
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            interpolated.append((x, y))
    
    return interpolated


def smooth_path(waypoints: List[Tuple[float, float]], 
               smoothing_factor: float = 0.3) -> List[Tuple[float, float]]:
    """Apply smoothing to a path of waypoints."""
    if len(waypoints) < 3:
        return waypoints.copy()
    
    smoothed = [waypoints[0]]  # Keep first point
    
    for i in range(1, len(waypoints) - 1):
        prev_point = waypoints[i-1]
        curr_point = waypoints[i]
        next_point = waypoints[i+1]
        
        # Calculate smooth position
        smooth_x = (1 - smoothing_factor) * curr_point[0] + \
                  smoothing_factor * 0.5 * (prev_point[0] + next_point[0])
        smooth_y = (1 - smoothing_factor) * curr_point[1] + \
                  smoothing_factor * 0.5 * (prev_point[1] + next_point[1])
        
        smoothed.append((smooth_x, smooth_y))
    
    smoothed.append(waypoints[-1])  # Keep last point
    return smoothed


def calculate_path_length(waypoints: List[Tuple[float, float]]) -> float:
    """Calculate total length of a path."""
    if len(waypoints) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(1, len(waypoints)):
        total_length += calculate_distance_2d(waypoints[i-1], waypoints[i])
    
    return total_length


def find_closest_point_on_path(point: Tuple[float, float],
                              path: List[Tuple[float, float]]) -> Tuple[int, float]:
    """Find closest point on path and return segment index and distance."""
    if not path:
        return -1, float('inf')
    
    min_distance = float('inf')
    closest_segment = -1
    
    for i in range(len(path)):
        distance = calculate_distance_2d(point, path[i])
        if distance < min_distance:
            min_distance = distance
            closest_segment = i
    
    return closest_segment, min_distance


def point_in_polygon(point: Tuple[float, float], 
                    polygon: List[Tuple[float, float]]) -> bool:
    """Check if point is inside polygon using ray casting algorithm."""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


class CoordinateTransformer:
    """Advanced coordinate transformation utilities."""
    
    def __init__(self, origin: Tuple[float, float] = (0, 0), 
                 scale: float = 1.0, rotation: float = 0.0):
        self.origin = origin
        self.scale = scale
        self.rotation = rotation
        
        # Pre-calculate transformation matrix components
        self.cos_rot = math.cos(rotation)
        self.sin_rot = math.sin(rotation)
    
    def transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Transform point using the configured transformation."""
        # Translate to origin
        x = (point[0] - self.origin[0]) * self.scale
        y = (point[1] - self.origin[1]) * self.scale
        
        # Apply rotation
        x_rot = x * self.cos_rot - y * self.sin_rot
        y_rot = x * self.sin_rot + y * self.cos_rot
        
        return (x_rot, y_rot)
    
    def inverse_transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Apply inverse transformation to point."""
        x, y = point
        
        # Apply inverse rotation
        x_rot = x * self.cos_rot + y * self.sin_rot
        y_rot = -x * self.sin_rot + y * self.cos_rot
        
        # Apply inverse scale and translation
        x_final = x_rot / self.scale + self.origin[0]
        y_final = y_rot / self.scale + self.origin[1]
        
        return (x_final, y_final)
    
    def transform_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Transform entire path."""
        return [self.transform_point(point) for point in path]
    
    def set_transformation(self, origin: Tuple[float, float] = None, 
                          scale: float = None, rotation: float = None):
        """Update transformation parameters."""
        if origin is not None:
            self.origin = origin
        if scale is not None:
            self.scale = scale
        if rotation is not None:
            self.rotation = rotation
            self.cos_rot = math.cos(rotation)
            self.sin_rot = math.sin(rotation)


def validate_coordinate(coord: Tuple[float, float], 
                       bounds: Optional[Tuple[Tuple[float, float], 
                                           Tuple[float, float]]] = None) -> bool:
    """Validate coordinate is within bounds and is valid."""
    x, y = coord
    
    # Check for NaN or infinite values
    if not (math.isfinite(x) and math.isfinite(y)):
        return False
    
    # Check bounds if provided
    if bounds:
        (min_x, min_y), (max_x, max_y) = bounds
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            return False
    
    return True


def snap_to_grid(point: Tuple[float, float], 
                grid_size: float = 1.0) -> Tuple[float, float]:
    """Snap point to nearest grid intersection."""
    x = round(point[0] / grid_size) * grid_size
    y = round(point[1] / grid_size) * grid_size
    return (x, y)