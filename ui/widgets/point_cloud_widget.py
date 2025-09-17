"""
Point Cloud Visualization Widget
Optimized widget for smooth point cloud display with drone visualization
"""

import math
import numpy as np
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPixmap

from config.settings import POINTCLOUD_CONFIG, UI_CONFIG, ASSET_PATHS
from core.waypoint_manager import WaypointManager


class SmoothPointCloudWidget(QWidget):
    """Optimized widget for smooth point cloud display."""
    
    waypoint_added = pyqtSignal()
    waypoint_changed = pyqtSignal()
    marker_changed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 200)
        self.setMouseTracking(True)
        
        # Initialize waypoint manager
        self.waypoint_manager = WaypointManager()
        
        # View parameters
        self.zoom = UI_CONFIG['default_zoom']
        self.pan_x = 0.0
        self.pan_y = 0.0
        
        # 3D view parameters
        self.rotation_x = 45.0  # Pitch (around X-axis)
        self.rotation_z = 0.0   # Yaw (around Z-axis)
        self.top_down_mode = True
        
        # Current frame data
        self.current_points = np.array([]).reshape(0, 3)
        self.current_colors = np.array([]).reshape(0, 3)
        self.projected_points = np.array([]).reshape(0, 2)
        
        # Raw data storage
        self.raw_points = np.array([]).reshape(0, 3)
        self.raw_min_height = 0.0
        self.raw_max_height = 5.0
        
        # Filtering options
        self.enable_z_filter = True
        self.max_z = POINTCLOUD_CONFIG['default_max_height']
        
        # Performance parameters
        self.max_points_render = POINTCLOUD_CONFIG['max_points_render']
        self.point_size = POINTCLOUD_CONFIG['point_size']
        self.dirty_projection = True
        
        # Grid and interaction
        self.grid_size = UI_CONFIG['grid_size']
        self.show_grid = True
        self.last_mouse_pos = None
        self.middle_dragging = False
        self.left_dragging = False
        
        # Height range for color mapping
        self.color_min_height = 0.0
        self.color_max_height = 5.0
        
        # Enhanced marker with parameters
        self.marker = None
        
        # Edit mode
        self.edit_mode = False
        
        # Performance tracking
        self.visible_points = 0
        
        # Drone display
        self.drone_image = None
        self.drone_position = None
        self.drone_orientation = 0.0
        self.drone_visible = False
        self.load_drone_image()
        
        # Load waypoints from file
        self.load_waypoints_from_json()
    
    def load_drone_image(self):
        """Load drone PNG image."""
        try:
            if ASSET_PATHS['drone_top'].exists():
                self.drone_image = QPixmap(str(ASSET_PATHS['drone_top']))
                if self.drone_image.isNull():
                    print("Failed to load drone image")
                    self.drone_image = None
        except Exception as e:
            print(f"Error loading drone image: {e}")
            self.drone_image = None
    
    def update_drone_data(self, position_x, position_y, yaw_radians):
        """Update drone position and orientation."""
        old_pos = self.drone_position
        old_orientation = self.drone_orientation
        
        self.drone_position = (position_x, position_y)
        self.drone_orientation = yaw_radians
        self.drone_visible = True
        
        # Only trigger repaint if something changed
        if (old_pos != self.drone_position or 
            abs(old_orientation - self.drone_orientation) > 0.01):
            self.update()
    
    def draw_drone(self, painter):
        """Draw drone at current position with orientation."""
        if not self.drone_position or not self.drone_image:
            return
            
        drone_x, drone_y = self.drone_position
        screen_x, screen_y = self.world_to_screen(-drone_y, drone_x)
        
        painter.save()
        
        try:
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.translate(screen_x, screen_y)
            painter.rotate(math.degrees(self.drone_orientation))
            
            drone_world_size = 0.6  # meters
            scaled_size = int(max(12, drone_world_size * self.zoom))
            
            drone_rect = self.rect()
            drone_rect.setSize(self.drone_image.size().scaled(scaled_size, scaled_size, Qt.KeepAspectRatio))
            drone_rect.moveCenter(self.rect().center())
            
            painter.drawPixmap(drone_rect, self.drone_image)
            
        finally:
            painter.restore()
    
    def set_z_filter(self, enabled=None, max_z=None):
        """Set Z height filtering options."""
        if enabled is not None:
            self.enable_z_filter = enabled
        if max_z is not None:
            self.max_z = max_z
            
        if len(self.raw_points) > 0:
            self.process_raw_points()
    
    def set_top_down_mode(self, enabled):
        """Toggle top-down view mode."""
        self.top_down_mode = enabled
        if enabled:
            self.rotation_x = 0.0
            self.rotation_z = 0.0
        else:
            self.rotation_x = 45.0
            self.rotation_z = 0.0
        self.dirty_projection = True
        self.update()
    
    def set_edit_mode(self, enabled):
        """Toggle edit mode."""
        self.edit_mode = enabled
        if not enabled:
            self.marker = None
            self.marker_changed.emit()
        self.update()
    
    def add_waypoint(self, x, y):
        """Add a new waypoint."""
        waypoint_id = self.waypoint_manager.add_waypoint(x, y)
        self.update()
        self.waypoint_changed.emit()
        return waypoint_id
    
    def update_waypoint(self, index, **kwargs):
        """Update waypoint properties."""
        success = self.waypoint_manager.update_waypoint(index, **kwargs)
        if success:
            self.update()
            self.waypoint_changed.emit()
        return success
    
    def delete_waypoint(self, index):
        """Delete a waypoint."""
        success = self.waypoint_manager.delete_waypoint(index)
        if success:
            self.update()
            self.waypoint_changed.emit()
        return success
    
    def get_waypoints(self):
        """Get all waypoints."""
        return self.waypoint_manager.get_all_waypoints()
    
    def create_marker(self, x, y):
        """Create a marker with default parameters."""
        self.marker = {
            'position': (x, y),
            'orientation': 0.0,
            'yaw_enable': False,
            'landing': False
        }
        self.marker_changed.emit()
        self.update()
    
    def update_marker(self, **kwargs):
        """Update marker properties."""
        if self.marker:
            self.marker.update(kwargs)
            self.marker_changed.emit()
            self.update()
    
    def get_marker(self):
        """Get marker data."""
        return self.marker
    
    def clear_marker(self):
        """Clear the marker."""
        self.marker = None
        self.marker_changed.emit()
        self.update()
    
    def save_waypoints_to_json(self):
        """Save waypoints to JSON file."""
        self.waypoint_manager.save_to_json()
    
    def load_waypoints_from_json(self):
        """Load waypoints from JSON file."""
        self.waypoint_manager.load_from_json()
        self.update()
    
    def process_raw_points(self):
        """Process raw points with minimal downsampling."""
        if len(self.raw_points) == 0:
            self.current_points = np.array([]).reshape(0, 3)
            self.current_colors = np.array([]).reshape(0, 3)
            self.dirty_projection = True
            self.update()
            return
        
        points = self.raw_points.copy()
        
        # Apply Z height filtering if enabled
        if self.enable_z_filter:
            mask = points[:, 2] <= self.max_z
            points = points[mask]
            
        if len(points) == 0:
            self.current_points = np.array([]).reshape(0, 3)
            self.current_colors = np.array([]).reshape(0, 3)
            self.dirty_projection = True
            self.update()
            return
        
        # Downsample if necessary for performance
        if len(points) > self.max_points_render:
            step = len(points) // self.max_points_render
            indices = np.arange(0, len(points), step)[:self.max_points_render]
            points = points[indices]
        
        self.current_points = points
        
        # Generate colors based on height
        if len(self.current_points) > 0:
            self.current_colors = self.generate_colors(self.current_points[:, 2])
        else:
            self.current_colors = np.array([]).reshape(0, 3)
        
        self.dirty_projection = True
        self.update()
    
    def update_current_frame(self, points):
        """Update with latest complete frame."""
        self.raw_points = points.copy() if len(points) > 0 else np.array([]).reshape(0, 3)
        
        if len(self.raw_points) > 0:
            heights = self.raw_points[:, 2]
            self.raw_min_height = np.min(heights)
            self.raw_max_height = np.max(heights)
            self.color_min_height = self.raw_min_height
            self.color_max_height = self.raw_max_height
        
        self.process_raw_points()
    
    def generate_colors(self, heights):
        """Generate red (low) to blue (high) gradient colors."""
        if len(heights) == 0:
            return np.array([]).reshape(0, 3)
        
        height_range = max(self.color_max_height - self.color_min_height, 0.1)
        normalized = (heights - self.color_min_height) / height_range
        normalized = np.clip(normalized, 0, 1)
        
        colors = np.zeros((len(heights), 3))
        
        # Red (low) to blue (high) gradient
        for i, h in enumerate(normalized):
            if h < 0.5:
                colors[i] = [1 - h * 2, h * 2, 0]
            else:
                colors[i] = [0, 1 - (h - 0.5) * 2, (h - 0.5) * 2]
        
        return colors
    
    def project_3d_to_2d(self, points_3d):
        """Project 3D points to 2D screen coordinates with rotation."""
        if len(points_3d) == 0:
            return np.array([]).reshape(0, 2)
        
        if self.top_down_mode:
            projected = np.column_stack([-points_3d[:, 1], points_3d[:, 0]])
        else:
            points = points_3d.copy()
            
            # Rotate around X-axis (pitch)
            if self.rotation_x != 0:
                angle_rad = math.radians(self.rotation_x)
                cos_x = math.cos(angle_rad)
                sin_x = math.sin(angle_rad)
                
                y_new = points[:, 1] * cos_x - points[:, 2] * sin_x
                z_new = points[:, 1] * sin_x + points[:, 2] * cos_x
                points[:, 1] = y_new
                points[:, 2] = z_new
            
            # Rotate around Z-axis (yaw)
            if self.rotation_z != 0:
                angle_rad = math.radians(self.rotation_z)
                cos_z = math.cos(angle_rad)
                sin_z = math.sin(angle_rad)
                
                x_new = points[:, 0] * cos_z - points[:, 1] * sin_z
                y_new = points[:, 0] * sin_z + points[:, 1] * cos_z
                points[:, 0] = x_new
                points[:, 1] = y_new
            
            projected = points[:, :2]
        
        return projected
    
    def world_to_screen(self, world_x, world_y):
        """Convert world coordinates to screen coordinates."""
        screen_x = (world_x - self.pan_x) * self.zoom + self.width() // 2
        screen_y = (self.pan_y - world_y) * self.zoom + self.height() // 2
        return screen_x, screen_y
    
    def screen_to_world(self, screen_x, screen_y):
        """Convert screen coordinates to world coordinates."""
        world_x = (screen_x - self.width() // 2) / self.zoom + self.pan_x
        world_y = self.pan_y - (screen_y - self.height() // 2) / self.zoom
        return world_x, world_y
    
    def paintEvent(self, event):
        """Paint the point cloud with performance optimizations."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        
        # Clear background
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        
        # Update projections if needed
        if self.dirty_projection and len(self.current_points) > 0:
            self.projected_points = self.project_3d_to_2d(self.current_points)
            self.dirty_projection = False
        
        # Draw grid
        if self.show_grid:
            self.draw_grid(painter)
        
        # Draw points
        self.draw_points(painter)
        
        # Draw markers/waypoints
        if self.edit_mode:
            self.draw_waypoints(painter)
        else:
            self.draw_marker(painter)
        
        # Draw drone
        if self.drone_visible and self.drone_position and self.drone_image:
            self.draw_drone(painter)
    
    def draw_grid(self, painter):
        """Draw grid with adaptive density."""
        painter.setPen(QPen(QColor(70, 70, 70), 1))
        
        # Calculate visible range
        half_width_world = self.width() / (2 * self.zoom)
        half_height_world = self.height() / (2 * self.zoom)
        
        min_x = self.pan_x - half_width_world
        max_x = self.pan_x + half_width_world
        min_y = self.pan_y - half_height_world
        max_y = self.pan_y + half_height_world
        
        # Adaptive grid spacing
        grid_spacing = self.grid_size
        while grid_spacing * self.zoom < 20:
            grid_spacing *= 2
        while grid_spacing * self.zoom > 100:
            grid_spacing /= 2
        
        # Draw vertical lines
        start_x = int(min_x / grid_spacing) * grid_spacing
        x = start_x
        while x <= max_x:
            screen_x, _ = self.world_to_screen(x, 0)
            if 0 <= screen_x <= self.width():
                painter.drawLine(int(screen_x), 0, int(screen_x), self.height())
            x += grid_spacing
        
        # Draw horizontal lines
        start_y = int(min_y / grid_spacing) * grid_spacing
        y = start_y
        while y <= max_y:
            _, screen_y = self.world_to_screen(0, y)
            if 0 <= screen_y <= self.height():
                painter.drawLine(0, int(screen_y), self.width(), int(screen_y))
            y += grid_spacing
    
    def draw_points(self, painter):
        """Draw point cloud."""
        if len(self.current_points) == 0 or len(self.projected_points) == 0:
            self.visible_points = 0
            return
        
        # Calculate visible bounds for culling
        margin = self.point_size * 2
        min_world_x, min_world_y = self.screen_to_world(-margin, self.height() + margin)
        max_world_x, max_world_y = self.screen_to_world(self.width() + margin, -margin)
        
        # Cull points outside view
        projected = self.projected_points
        colors = self.current_colors
        
        visible_mask = ((projected[:, 0] >= min_world_x) & (projected[:, 0] <= max_world_x) &
                       (projected[:, 1] >= min_world_y) & (projected[:, 1] <= max_world_y))
        
        if not np.any(visible_mask):
            self.visible_points = 0
            return
        
        visible_projected = projected[visible_mask]
        visible_colors = colors[visible_mask] if len(colors) > 0 else None
        self.visible_points = len(visible_projected)
        
        # Draw points
        painter.setPen(Qt.NoPen)
        
        # Convert to screen coordinates
        screen_coords = np.zeros_like(visible_projected)
        for i, point in enumerate(visible_projected):
            screen_coords[i] = self.world_to_screen(point[0], point[1])
        
        # Draw points with colors
        if visible_colors is not None:
            for coord, color in zip(screen_coords, visible_colors):
                x, y = int(coord[0]), int(coord[1])
                if (-self.point_size <= x <= self.width() + self.point_size and 
                    -self.point_size <= y <= self.height() + self.point_size):
                    qcolor = QColor(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                    painter.setBrush(QBrush(qcolor))
                    painter.drawEllipse(x - self.point_size//2, y - self.point_size//2, 
                                      self.point_size, self.point_size)
        else:
            painter.setBrush(QBrush(QColor(100, 150, 255)))
            for coord in screen_coords:
                x, y = int(coord[0]), int(coord[1])
                if (-self.point_size <= x <= self.width() + self.point_size and 
                    -self.point_size <= y <= self.height() + self.point_size):
                    painter.drawEllipse(x - self.point_size//2, y - self.point_size//2, 
                                      self.point_size, self.point_size)
    
    def draw_marker(self, painter):
        """Draw the enhanced marker if it exists."""
        if self.marker is None:
            return
            
        marker_x, marker_y = self.marker['position']
        screen_x, screen_y = self.world_to_screen(-marker_y, marker_x)
        
        # Draw marker circle
        painter.setBrush(QBrush(QColor(255, 255, 0, 200)))
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.drawEllipse(int(screen_x - 8), int(screen_y - 8), 16, 16)
        
        # Draw orientation arrow
        orientation = self.marker['orientation']
        arrow_length = 25
        arrow_x = screen_x + arrow_length * math.sin(-orientation)
        arrow_y = screen_y - arrow_length * math.cos(-orientation)
        
        painter.setPen(QPen(QColor(255, 0, 0), 3))
        painter.drawLine(int(screen_x), int(screen_y), int(arrow_x), int(arrow_y))
        
        # Draw arrowhead
        arrowhead_length = 8
        arrowhead_angle = 0.5
        
        arrowhead_x1 = arrow_x - arrowhead_length * math.sin(-orientation + arrowhead_angle)
        arrowhead_y1 = arrow_y + arrowhead_length * math.cos(-orientation + arrowhead_angle)
        arrowhead_x2 = arrow_x - arrowhead_length * math.sin(-orientation - arrowhead_angle)
        arrowhead_y2 = arrow_y + arrowhead_length * math.cos(-orientation - arrowhead_angle)
        
        painter.drawLine(int(arrow_x), int(arrow_y), int(arrowhead_x1), int(arrowhead_y1))
        painter.drawLine(int(arrow_x), int(arrow_y), int(arrowhead_x2), int(arrowhead_y2))
        
        # Draw coordinates
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        
        coord_text = f"({marker_x:.2f}, {marker_y:.2f})"
        painter.drawText(int(screen_x + 15), int(screen_y - 15), coord_text)
    
    def draw_waypoints(self, painter):
        """Draw waypoints in edit mode."""
        waypoints = self.waypoint_manager.get_all_waypoints()
        
        for i, waypoint in enumerate(waypoints):
            pos_x, pos_y = waypoint['position']
            screen_x, screen_y = self.world_to_screen(-pos_y, pos_x)
            
            # Draw waypoint circle
            color = QColor(0, 255, 0, 200) if waypoint['added'] else QColor(255, 255, 0, 200)
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.drawEllipse(int(screen_x - 8), int(screen_y - 8), 16, 16)
            
            # Draw waypoint number
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.drawText(int(screen_x - 4), int(screen_y + 3), str(i + 1))
            
            # Draw orientation arrow if waypoint is added
            if waypoint['added']:
                orientation = waypoint['orientation']
                arrow_length = 20
                arrow_x = screen_x + arrow_length * math.sin(-orientation)
                arrow_y = screen_y - arrow_length * math.cos(-orientation)
                
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.drawLine(int(screen_x), int(screen_y), int(arrow_x), int(arrow_y))
    
    def mousePressEvent(self, event):
        """Handle mouse press."""
        self.last_mouse_pos = event.pos()
        
        if event.button() == Qt.MiddleButton:
            self.middle_dragging = True
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton and not self.top_down_mode:
            self.left_dragging = True
            self.setCursor(Qt.SizeAllCursor)
        elif event.button() == Qt.RightButton:
            if self.top_down_mode:
                screen_pos = self.screen_to_world(event.x(), event.y())
                world_pos = (screen_pos[1], -screen_pos[0])
                
                if self.edit_mode:
                    waypoint_index = self.add_waypoint(world_pos[0], world_pos[1])
                    self.waypoint_added.emit()
                    print(f"Waypoint {waypoint_index + 1} placed: ({world_pos[0]:.2f}, {world_pos[1]:.2f})")
                else:
                    self.create_marker(world_pos[0], world_pos[1])
                    print(f"Marker placed: ({world_pos[0]:.2f}, {world_pos[1]:.2f})")
                
                self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move."""
        if self.last_mouse_pos:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()
            
            if self.middle_dragging:
                self.pan_x -= dx / self.zoom
                self.pan_y += dy / self.zoom
                self.update()
            
            elif self.left_dragging and not self.top_down_mode:
                self.rotation_z += dx * 0.5
                self.rotation_x += dy * 0.5
                
                self.rotation_x = max(-90, min(90, self.rotation_x))
                self.rotation_z = self.rotation_z % 360
                
                self.dirty_projection = True
                self.update()
        
        self.last_mouse_pos = event.pos()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MiddleButton:
            self.middle_dragging = False
        elif event.button() == Qt.LeftButton:
            self.left_dragging = False
        
        self.setCursor(Qt.ArrowCursor)
    
    def wheelEvent(self, event):
        """Handle zoom."""
        mouse_world_before = self.screen_to_world(event.x(), event.y())
        
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1/1.15
        self.zoom *= zoom_factor
        self.zoom = max(UI_CONFIG['zoom_limits'][0], min(UI_CONFIG['zoom_limits'][1], self.zoom))
        
        mouse_world_after = self.screen_to_world(event.x(), event.y())
        self.pan_x += mouse_world_before[0] - mouse_world_after[0]
        self.pan_y += mouse_world_before[1] - mouse_world_after[1]
        
        self.update()