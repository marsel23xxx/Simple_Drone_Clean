#!/usr/bin/env python3
"""
Hybrid Point Cloud Viewer - WebSocket Commands + TCP Point Data
- TCP server for receiving point cloud data (port 12345)
- WebSocket client for sending drone commands (192.168.1.88:9000)
- Middle mouse to pan, left click for 3D rotation
- Right click places single marker with coordinates (or waypoints in Edit Mode)
- Red (low) to blue (high) height gradient
- Top-down view toggle
- Frame skipping for performance optimization
- Edit Mode with waypoint management and joystick orientation control
- JSON persistence for waypoint coordinates
- Enhanced marker with orientation, yaw enable, and landing parameters
"""
import sys
import struct
import numpy as np
import threading
import time
import open3d as o3d
import math
import json
import os
import asyncio
import websockets
import socket
from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QGroupBox, QCheckBox,
                            QDoubleSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
                            QDialog, QDialogButtonBox)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject, QThread, QRect
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPixmap

# Performance Configuration
PROCESS_FRAME_SKIP = 4      # Process every Nth frame (1 = no skipping)
DISPLAY_FRAME_SKIP = 1     # Display every Nth processed frame

# TCP Configuration for Point Cloud Data
TCP_LISTEN_PORT = 12345

# WebSocket Configuration for Commands
WEBSOCKET_IP = "192.168.1.88"
WEBSOCKET_PORT = 9000

# JSON file for storing coordinates
COORDINATES_FILE = "coordinates.json"


class TCPDataReceiver(QObject):
    """TCP server for receiving point cloud data."""
    
    frame_received = pyqtSignal(np.ndarray)
    connection_status = pyqtSignal(bool, str)  # connected, message
    
    def __init__(self):
        super().__init__()
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.connected = False
        
        # Frame counters
        self.received_frame_count = 0
        self.processed_frame_count = 0
        self.displayed_frame_count = 0
        
        # Frame skipping configuration
        self.process_frame_skip = PROCESS_FRAME_SKIP
        self.display_frame_skip = DISPLAY_FRAME_SKIP
    
    def start_server(self):
        """Start TCP server to listen for point cloud data."""
        self.running = True
        
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('', TCP_LISTEN_PORT))
            self.server_socket.listen(1)
            self.server_socket.settimeout(1.0)  # 1 second timeout for accept
            
            self.connection_status.emit(False, f"TCP server listening on port {TCP_LISTEN_PORT}")
            print(f"TCP server listening on port {TCP_LISTEN_PORT}")
            
            while self.running:
                try:
                    print("Waiting for point cloud data connection...")
                    self.client_socket, client_address = self.server_socket.accept()
                    self.client_socket.settimeout(2.0)  # 2 second timeout for recv
                    
                    self.connected = True
                    self.connection_status.emit(True, f"Point cloud source connected from {client_address[0]}")
                    print(f"Point cloud source connected from {client_address}")
                    
                    # Handle client connection
                    self.handle_client_connection()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"TCP accept error: {e}")
                        self.connection_status.emit(False, f"Accept error: {e}")
                
                if self.client_socket:
                    self.client_socket.close()
                    self.client_socket = None
                    self.connected = False
                    self.connection_status.emit(False, "Point cloud source disconnected")
                
        except Exception as e:
            print(f"TCP server error: {e}")
            self.connection_status.emit(False, f"Server error: {e}")
        finally:
            self.cleanup_server()
    
    def handle_client_connection(self):
        """Handle data from connected client."""
        buffer = b''
        
        while self.running and self.connected:
            try:
                data = self.client_socket.recv(8192)
                if not data:
                    print("Client disconnected")
                    break
                
                buffer += data
                
                # Process complete messages in buffer
                while len(buffer) >= 4:
                    # Read points count
                    points_count = struct.unpack('<I', buffer[:4])[0]
                    expected_size = 4 + points_count * 3 * 4  # header + points data
                    
                    if len(buffer) >= expected_size:
                        # Extract complete message
                        message = buffer[:expected_size]
                        buffer = buffer[expected_size:]
                        
                        # Process the point cloud data
                        self.process_binary_data(message)
                    else:
                        # Wait for more data
                        break
                        
            except socket.timeout:
                continue
            except Exception as e:
                print(f"TCP receive error: {e}")
                break
    
    def process_binary_data(self, data):
        """Process binary point cloud data."""
        try:
            # Assume data format: [points_count (4 bytes)] + [points_data]
            if len(data) < 4:
                return
            
            points_count = struct.unpack('<I', data[:4])[0]
            points_data = data[4:]
            
            expected_size = points_count * 3 * 4  # 3 floats per point
            if len(points_data) != expected_size:
                print(f"Data size mismatch: expected {expected_size}, got {len(points_data)}")
                return
            
            points = np.frombuffer(points_data, dtype=np.float32).reshape(-1, 3)
            
            self.received_frame_count += 1
            
            # Process frame skip logic
            should_process = (self.received_frame_count % self.process_frame_skip) == 0
            
            if should_process:
                self.processed_frame_count += 1
                
                # Display frame skip logic
                should_display = (self.processed_frame_count % self.display_frame_skip) == 0
                
                if should_display:
                    self.displayed_frame_count += 1
                    self.frame_received.emit(points)
                    
        except Exception as e:
            print(f"Error processing binary data: {e}")
    
    def cleanup_server(self):
        """Clean up server resources."""
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
        
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
        
        self.connected = False
        self.connection_status.emit(False, "TCP server stopped")
    
    def stop_server(self):
        """Stop the TCP server."""
        self.running = False


class WebSocketCommandClient(QObject):
    """WebSocket client for sending commands only."""
    
    connection_status = pyqtSignal(bool, str)  # connected, message
    
    def __init__(self):
        super().__init__()
        self.websocket = None
        self.running = False
        self.connected = False
    
    async def connect_and_maintain(self):
        """Connect to WebSocket server and maintain connection."""
        uri = f"ws://{WEBSOCKET_IP}:{WEBSOCKET_PORT}"
        
        try:
            print(f"Connecting to command WebSocket {uri}...")
            self.websocket = await websockets.connect(uri)
            self.connected = True
            self.connection_status.emit(True, f"Command WebSocket connected")
            print(f"Connected to command WebSocket {uri}")
            
            # Keep connection alive
            while self.running and self.connected:
                try:
                    await asyncio.sleep(10)
                    if self.websocket:
                        await self.websocket.ping()
                    else:
                        break
                except Exception as e:
                    print(f"WebSocket ping error: {e}")
                    break
                    
        except Exception as e:
            print(f"WebSocket connection error: {e}")
            self.connected = False
            self.connection_status.emit(False, f"Connection error: {e}")
        finally:
            if self.websocket:
                await self.websocket.close()
            self.connected = False
            self.connection_status.emit(False, "Disconnected")
    
    async def send_command(self, command):
        """Send command through WebSocket."""
        if self.websocket and self.connected:
            try:
                await self.websocket.send(command)
                print(f"Sent command: {command}")
                return True
            except Exception as e:
                print(f"Error sending command: {e}")
                self.connected = False
                self.connection_status.emit(False, f"Send error: {e}")
                return False
        else:
            print("WebSocket not connected, cannot send command")
            return False
    
    def start_client(self):
        """Start the WebSocket client."""
        self.running = True
        
    def stop_client(self):
        """Stop the WebSocket client."""
        self.running = False


class TCPServerThread(QThread):
    """Thread to run TCP server."""
    
    def __init__(self, tcp_receiver):
        super().__init__()
        self.tcp_receiver = tcp_receiver
    
    def run(self):
        """Run the TCP server."""
        self.tcp_receiver.start_server()


class WebSocketCommandThread(QThread):
    """Thread to run WebSocket command client."""
    
    def __init__(self, websocket_client):
        super().__init__()
        self.websocket_client = websocket_client
        self.loop = None
    
    def run(self):
        """Run the WebSocket client in async event loop."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self.websocket_client.connect_and_maintain())
        except Exception as e:
            print(f"WebSocket thread error: {e}")
        finally:
            self.loop.close()
    
    def send_command_sync(self, command):
        """Send command synchronously from main thread."""
        if self.loop and not self.loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(
                self.websocket_client.send_command(command), self.loop
            )
            try:
                return future.result(timeout=2.0)  # 2 second timeout
            except Exception as e:
                print(f"Error sending command sync: {e}")
                return False
        return False


class JoystickDialog(QDialog):
    """Dialog with joystick control for orientation editing."""
    
    def __init__(self, current_orientation=0.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Orientation")
        self.setFixedSize(300, 350)
        self.setModal(True)
        
        self.orientation = current_orientation
        self.joystick_center = (150, 150)
        self.joystick_radius = 80
        self.knob_pos = self.joystick_center
        self.dragging = False
        
        # Set initial knob position based on current orientation
        self.set_knob_from_orientation()
        
        layout = QVBoxLayout(self)
        
        # Orientation display
        self.orientation_label = QLabel(f"Orientation: {self.orientation:.3f} rad")
        self.orientation_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.orientation_label)
        
        # Joystick area (will be drawn in paintEvent)
        layout.addStretch()
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def set_knob_from_orientation(self):
        """Set knob position based on orientation value."""
        # Up is 0, anticlockwise positive
        angle = -self.orientation
        x = self.joystick_center[0] + self.joystick_radius * 0.8 * math.sin(angle)
        y = self.joystick_center[1] + self.joystick_radius * 0.8 * math.cos(angle)
        self.knob_pos = (x, y)
    
    def update_orientation_from_knob(self):
        """Update orientation based on knob position."""
        dx = self.knob_pos[0] - self.joystick_center[0]
        dy = self.knob_pos[1] - self.joystick_center[1]
        
        if dx == 0 and dy == 0:
            self.orientation = 0.0
        else:
            # Calculate angle: up is 0, anticlockwise positive
            angle = math.atan2(dx, -dy)  # atan2(x, y) because up is 0
            self.orientation = -angle
        
        self.orientation_label.setText(f"Orientation: {self.orientation:.3f} rad")
    
    def paintEvent(self, event):
        """Draw the joystick."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw joystick base
        painter.setBrush(QBrush(QColor(60, 60, 60)))
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        painter.drawEllipse(self.joystick_center[0] - self.joystick_radius,
                          self.joystick_center[1] - self.joystick_radius,
                          self.joystick_radius * 2, self.joystick_radius * 2)
        
        # Draw direction indicators
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(self.joystick_center[0] - 5, self.joystick_center[1] - self.joystick_radius - 10, "0°")
        painter.drawText(self.joystick_center[0] + self.joystick_radius + 10, self.joystick_center[1] + 5, "90°")
        painter.drawText(self.joystick_center[0] - 5, self.joystick_center[1] + self.joystick_radius + 20, "180°")
        painter.drawText(self.joystick_center[0] - self.joystick_radius - 25, self.joystick_center[1] + 5, "-90°")
        
        # Draw knob
        knob_radius = 15
        painter.setBrush(QBrush(QColor(100, 150, 255)))
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawEllipse(int(self.knob_pos[0] - knob_radius),
                          int(self.knob_pos[1] - knob_radius),
                          knob_radius * 2, knob_radius * 2)
    
    def mousePressEvent(self, event):
        """Handle mouse press on joystick."""
        if event.button() == Qt.LeftButton:
            knob_radius = 15
            dx = event.x() - self.knob_pos[0]
            dy = event.y() - self.knob_pos[1]
            if dx*dx + dy*dy <= knob_radius*knob_radius:
                self.dragging = True
    
    def mouseMoveEvent(self, event):
        """Handle knob dragging."""
        if self.dragging:
            # Constrain knob to joystick area
            dx = event.x() - self.joystick_center[0]
            dy = event.y() - self.joystick_center[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance <= self.joystick_radius:
                self.knob_pos = (event.x(), event.y())
            else:
                # Constrain to circle edge
                factor = self.joystick_radius / distance
                self.knob_pos = (self.joystick_center[0] + dx * factor,
                               self.joystick_center[1] + dy * factor)
            
            self.update_orientation_from_knob()
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self.dragging = False


class SmoothPointCloudWidget(QWidget):
    """Optimized widget for smooth point cloud display."""
    waypoint_added = pyqtSignal()
    waypoint_changed = pyqtSignal()  # Signal for waypoint changes
    marker_changed = pyqtSignal()  # Signal for marker changes
    
    def __init__(self):
        # Drone display
        self.drone_image = None
        self.drone_position = None  # (x, y) in world coordinates  
        self.drone_orientation = 0.0  # yaw in radians
        self.drone_visible = False
        self.load_drone_image()
        
        super().__init__()
        self.setMinimumSize(800, 600)
        self.setMouseTracking(True)
        
        # View parameters
        self.zoom = 20.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        
        # 3D view parameters
        self.rotation_x = 45.0  # Pitch (around X-axis)
        self.rotation_z = 0.0   # Yaw (around Z-axis)
        self.top_down_mode = True
        
        # Current frame data
        self.current_points = np.array([]).reshape(0, 3)
        self.current_colors = np.array([]).reshape(0, 3)
        self.projected_points = np.array([]).reshape(0, 2)  # Cache projected points
        
        # Raw data storage
        self.raw_points = np.array([]).reshape(0, 3)
        self.raw_min_height = 0.0
        self.raw_max_height = 5.0
        
        # Filtering options
        self.enable_z_filter = True
        self.max_z = 1.5
        
        # Performance parameters
        self.max_points_render = 100000
        self.point_size = 3
        self.dirty_projection = True
        
        # Grid and interaction
        self.grid_size = 1.0
        self.show_grid = True
        self.last_mouse_pos = None
        self.middle_dragging = False
        self.left_dragging = False
        
        # Height range for color mapping (uses raw data range)
        self.color_min_height = 0.0
        self.color_max_height = 5.0
        
        # Enhanced marker with parameters (for non-edit mode)
        self.marker = None  # Dictionary with position, orientation, yaw_enable, landing
        
        # Edit mode and waypoints
        self.edit_mode = False
        self.waypoints = []  # List of waypoint dictionaries
        
        # Performance tracking
        self.visible_points = 0
        
        # Load waypoints from JSON on startup
        self.load_waypoints_from_json()
    
    def load_drone_image(self):
        """Load drone PNG image."""
        try:
            from PyQt5.QtGui import QPixmap
            self.drone_image = QPixmap("DRONETOP.png")  # Your PNG file path
            if self.drone_image.isNull():
                print("Failed to load drone image")
                self.drone_image = None
        except Exception as e:
            print(f"Error loading drone image: {e}")
            self.drone_image = None
    
    def update_drone_data(self, position_x, position_y, yaw_radians):
        """Update drone position and orientation."""
        # Check if data actually changed
        old_pos = self.drone_position
        old_orientation = self.drone_orientation
        
        self.drone_position = (position_x, position_y)
        self.drone_orientation = yaw_radians
        self.drone_visible = True
        
        # Only trigger repaint if something changed
        if (old_pos != self.drone_position or 
            abs(old_orientation - self.drone_orientation) > 0.01):
            self.update()  # Trigger repaint
    
    def draw_drone(self, painter):
        """Draw drone at current position with orientation."""
        if not self.drone_position or not self.drone_image:
            return
            
        drone_x, drone_y = self.drone_position
        screen_x, screen_y = self.world_to_screen(-drone_y, drone_x)
        
        # Save painter state
        painter.save()
        
        try:
            # Improve rendering quality for small images
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            
            # Move to drone position and rotate
            painter.translate(screen_x, screen_y)
            painter.rotate(math.degrees(self.drone_orientation))
            
            # Calculate drone size with larger minimum
            drone_world_size = 0.6  # meters
            scaled_size = int(max(12, drone_world_size * self.zoom))  # Larger minimum
            
            # Draw drone image centered
            drone_rect = QRect(-scaled_size//2, -scaled_size//2, scaled_size, scaled_size)
            painter.drawPixmap(drone_rect, self.drone_image)
            
        finally:
            # Always restore painter state
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
            # Clear single marker when entering edit mode
            self.marker = None
            self.marker_changed.emit()
        self.update()
    
    def add_waypoint(self, x, y):
        """Add a new waypoint."""
        waypoint = {
            'position': (x, y),
            'orientation': 0.0,
            'yaw_enable': False,
            'landing': False,
            'added': False
        }
        self.waypoints.append(waypoint)
        self.update()
        self.waypoint_changed.emit()  # Emit change signal
        return len(self.waypoints) - 1  # Return index
    
    def update_waypoint(self, index, **kwargs):
        """Update waypoint properties."""
        if 0 <= index < len(self.waypoints):
            self.waypoints[index].update(kwargs)
            self.update()
            # Only save to JSON if waypoint is added (not for temporary edits)
            if self.waypoints[index]['added']:
                self.save_waypoints_to_json()
            self.waypoint_changed.emit()  # Emit change signal
    
    def delete_waypoint(self, index):
        """Delete a waypoint."""
        if 0 <= index < len(self.waypoints):
            del self.waypoints[index]
            self.update()
            self.save_waypoints_to_json()  # Save after deletion
            self.waypoint_changed.emit()  # Emit change signal
    
    def get_waypoints(self):
        """Get all waypoints."""
        return self.waypoints.copy()
    
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
    
    def save_waypoints_to_json(self):
        """Save added waypoints to JSON file."""
        try:
            # Filter only added waypoints
            added_waypoints = [wp for wp in self.waypoints if wp['added']]
            
            # Convert to the requested format: [x, y, orientation, yaw_enable, landing]
            coordinates_array = []
            for wp in added_waypoints:
                x, y = wp['position']
                coordinates_array.append([
                    x,
                    y,
                    wp['orientation'],
                    1 if wp['yaw_enable'] else 0,
                    1 if wp['landing'] else 0
                ])
            
            # Save to JSON file
            with open(COORDINATES_FILE, 'w') as f:
                json.dump(coordinates_array, f, indent=2)
            
            print(f"Saved {len(coordinates_array)} waypoints to {COORDINATES_FILE}")
            
        except Exception as e:
            print(f"Error saving waypoints to JSON: {e}")
    
    def load_waypoints_from_json(self):
        """Load waypoints from JSON file on startup."""
        try:
            if os.path.exists(COORDINATES_FILE):
                with open(COORDINATES_FILE, 'r') as f:
                    coordinates_array = json.load(f)
                
                # Clear existing waypoints
                self.waypoints = []
                
                # Convert from JSON format to waypoint objects
                for coord in coordinates_array:
                    if len(coord) >= 5:  # Ensure we have all required fields
                        waypoint = {
                            'position': (coord[0], coord[1]),
                            'orientation': coord[2],
                            'yaw_enable': bool(coord[3]),
                            'landing': bool(coord[4]),
                            'added': True  # All loaded waypoints are considered added
                        }
                        self.waypoints.append(waypoint)
                
                print(f"Loaded {len(self.waypoints)} waypoints from {COORDINATES_FILE}")
                self.update()
                
        except Exception as e:
            print(f"Error loading waypoints from JSON: {e}")
            self.waypoints = []  # Reset to empty list on error
        
    def process_raw_points(self):
        """Process raw points with minimal downsampling for smooth display."""
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
        
        # Generate colors based on height using raw data range
        if len(self.current_points) > 0:
            self.current_colors = self.generate_colors(self.current_points[:, 2])
        else:
            self.current_colors = np.array([]).reshape(0, 3)
        
        self.dirty_projection = True
        self.update()
        
    def update_current_frame(self, points):
        """Update with latest complete frame."""
        # Store raw points
        self.raw_points = points.copy() if len(points) > 0 else np.array([]).reshape(0, 3)
        
        if len(self.raw_points) > 0:
            heights = self.raw_points[:, 2]
            self.raw_min_height = np.min(heights)
            self.raw_max_height = np.max(heights)
            self.color_min_height = self.raw_min_height
            self.color_max_height = self.raw_max_height
        
        # Process the points
        self.process_raw_points()
    
    def generate_colors(self, heights):
        """Generate red (low) to blue (high) gradient colors using fixed color range."""
        if len(heights) == 0:
            return np.array([]).reshape(0, 3)
        
        height_range = max(self.color_max_height - self.color_min_height, 0.1)
        normalized = (heights - self.color_min_height) / height_range
        normalized = np.clip(normalized, 0, 1)
        
        colors = np.zeros((len(heights), 3))
        
        # Red (low) to blue (high) gradient through green
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
            
        # Draw drone (add this at the end of paintEvent, after drawing points/waypoints)
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
        """Draw the enhanced marker if it exists (non-edit mode)."""
        if self.marker is None:
            return
            
        marker_x, marker_y = self.marker['position']
        screen_x, screen_y = self.world_to_screen(-marker_y, marker_x)
        
        # Draw marker as circle
        painter.setBrush(QBrush(QColor(255, 255, 0, 200)))  # Yellow marker
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.drawEllipse(int(screen_x - 8), int(screen_y - 8), 16, 16)
        
        # Draw orientation arrow
        orientation = self.marker['orientation']
        arrow_length = 25
        arrow_x = screen_x + arrow_length * math.sin(-orientation)
        arrow_y = screen_y - arrow_length * math.cos(-orientation)  # Negative because screen Y is inverted
        
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
        
        # Draw coordinates and parameters
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        
        coord_text = f"({marker_x:.2f}, {marker_y:.2f})"
        painter.drawText(int(screen_x + 15), int(screen_y - 15), coord_text)
        
        # Draw parameter indicators
        y_offset = -35
        painter.setFont(QFont("Arial", 8))
        
        if self.marker['yaw_enable']:
            painter.setPen(QPen(QColor(0, 255, 0), 1))
            painter.drawText(int(screen_x + 15), int(screen_y + y_offset), "YAW")
            y_offset -= 15
        
        if self.marker['landing']:
            painter.setPen(QPen(QColor(255, 100, 0), 1))
            painter.drawText(int(screen_x + 15), int(screen_y + y_offset), "LAND")
    
    def draw_waypoints(self, painter):
        """Draw waypoints in edit mode."""
        for i, waypoint in enumerate(self.waypoints):
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
                arrow_y = screen_y - arrow_length * math.cos(-orientation)  # Negative because screen Y is inverted
                
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.drawLine(int(screen_x), int(screen_y), int(arrow_x), int(arrow_y))
                
                # Draw arrowhead
                arrowhead_length = 6
                arrowhead_angle = 0.5
                
                arrowhead_x1 = arrow_x - arrowhead_length * math.sin(-orientation + arrowhead_angle)
                arrowhead_y1 = arrow_y + arrowhead_length * math.cos(-orientation + arrowhead_angle)
                arrowhead_x2 = arrow_x - arrowhead_length * math.sin(-orientation - arrowhead_angle)
                arrowhead_y2 = arrow_y + arrowhead_length * math.cos(-orientation - arrowhead_angle)
                
                painter.drawLine(int(arrow_x), int(arrow_y), int(arrowhead_x1), int(arrowhead_y1))
                painter.drawLine(int(arrow_x), int(arrow_y), int(arrowhead_x2), int(arrowhead_y2))
    
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
                # Place marker
                screen_pos = self.screen_to_world(event.x(), event.y())
                world_pos = (screen_pos[1], -screen_pos[0])  # Inverse of (-y, x)
                
                if self.edit_mode:
                    # Add waypoint in edit mode
                    waypoint_index = self.add_waypoint(world_pos[0], world_pos[1])
                    self.waypoint_added.emit()
                    print(f"Waypoint {waypoint_index + 1} placed: ({world_pos[0]:.2f}, {world_pos[1]:.2f})")
                else:
                    # Create enhanced marker in normal mode
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
        self.zoom = max(0.5, min(500.0, self.zoom))
        
        mouse_world_after = self.screen_to_world(event.x(), event.y())
        self.pan_x += mouse_world_before[0] - mouse_world_after[0]
        self.pan_y += mouse_world_before[1] - mouse_world_after[1]
        
        self.update()

    def get_marker_coordinates(self):
        """Get marker coordinates or (0, 0) if no marker."""
        if self.marker is None:
            return (0.0, 0.0)
        return self.marker['position']

    def clear_marker(self):
        """Clear the marker."""
        self.marker = None
        self.marker_changed.emit()
        self.update()


class HybridPointCloudWindow(QMainWindow):
    """Main window with TCP point cloud receiver and WebSocket command sender."""
    
    def __init__(self):
        # Add drone parser
        from DataParserLib import DroneParser  # Adjust import path
        self.drone_parser = DroneParser(port=8889)  # or whatever port you use
        self.drone_parser.start()
        
        # Drone data update timer
        self.drone_timer = QTimer()
        self.drone_timer.timeout.connect(self.update_drone_data)
        self.drone_timer.start(0)  # Update every 100ms
        
        super().__init__()
        self.setWindowTitle("Hybrid Point Cloud Viewer - TCP Data + WebSocket Commands")
        self.setGeometry(100, 100, 1800, 1000)
        
        # Frame tracking
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0.0
        
        # TCP server for point cloud data
        self.tcp_receiver = TCPDataReceiver()
        self.tcp_thread = None
        
        # WebSocket client for commands
        self.websocket_client = WebSocketCommandClient()
        self.websocket_thread = None
        self.start_enabled = False  # Track start/pause state
        
        # Setup UI
        self.setup_ui()
        
        # Connect signals
        self.tcp_receiver.frame_received.connect(self.display_frame)
        self.tcp_receiver.connection_status.connect(self.update_tcp_status)
        self.websocket_client.connection_status.connect(self.update_websocket_status)
        
        # Start both servers/clients
        self.start_tcp_server()
        
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)
    
    def update_drone_data(self):
        """Update drone position from parser."""
        try:
            # Get latest drone data
            latest_data = self.drone_parser.get_latest()
            if not latest_data:
                return
                
            position = self.drone_parser.get_position(latest_data)
            rpy = self.drone_parser.get_rpy(latest_data)
            
            if position and rpy:
                # Check if data actually changed before updating
                new_x, new_y, new_yaw = position['x'], position['y'], rpy['yaw']
                
                # Only update if position/orientation changed significantly
                if (not hasattr(self, '_last_drone_data') or 
                    abs(new_x - self._last_drone_data[0]) > 0.01 or
                    abs(new_y - self._last_drone_data[1]) > 0.01 or
                    abs(new_yaw - self._last_drone_data[2]) > 0.01):
                    
                    self.map_widget.update_drone_data(new_x, new_y, new_yaw)
                    self._last_drone_data = (new_x, new_y, new_yaw)
                    
        except Exception as e:
            # Silently handle errors - drone might not be connected
            pass
    
    def start_tcp_server(self):
        """Start the TCP server thread."""
        if self.tcp_thread is None or not self.tcp_thread.isRunning():
            self.tcp_thread = TCPServerThread(self.tcp_receiver)
            self.tcp_thread.start()
    
    def start_websocket_client(self):
        """Start the WebSocket client thread."""
        if self.websocket_thread is None or not self.websocket_thread.isRunning():
            self.websocket_client.start_client()
            self.websocket_thread = WebSocketCommandThread(self.websocket_client)
            self.websocket_thread.start()
    
    def toggle_connection(self):
        """Toggle WebSocket connection."""
        if self.websocket_client.connected:
            self.disconnect_websocket()
        else:
            self.connect_websocket()

    def connect_websocket(self):
        """Connect to WebSocket server."""
        if self.websocket_thread is None or not self.websocket_thread.isRunning():
            self.start_websocket_client()

    def disconnect_websocket(self):
        """Disconnect from WebSocket server."""
        self.websocket_client.stop_client()
        if self.websocket_thread:
            self.websocket_thread.quit()
            self.websocket_thread.wait()
            self.websocket_thread = None
    
    def setup_ui(self):
        """Setup UI with dual connection status and command controls."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        
        # Map widget
        self.map_widget = SmoothPointCloudWidget()
        layout.addWidget(self.map_widget, stretch=3)
        
        # Control panel
        control_panel = QWidget()
        control_panel.setMaximumWidth(400)
        control_layout = QVBoxLayout(control_panel)
        
        # Connection Status
        connection_group = QGroupBox("Connection Status")
        connection_layout = QVBoxLayout(connection_group)
        
        self.tcp_label = QLabel("TCP Server: Starting...")
        connection_layout.addWidget(self.tcp_label)
        
        self.websocket_label = QLabel("WebSocket: Connecting...")
        connection_layout.addWidget(self.websocket_label)
        
        # Connect/Disconnect Button
        self.connect_button = QPushButton("Connect")
        self.connect_button.setEnabled(True)
        self.connect_button.clicked.connect(self.toggle_connection)
        connection_layout.addWidget(self.connect_button)
        
        control_layout.addWidget(connection_group)
        
        # Status
        self.status_label = QLabel("Status: Starting servers...")
        control_layout.addWidget(self.status_label)
        
        # WebSocket Command Controls
        command_group = QGroupBox("Drone Commands")
        command_layout = QVBoxLayout(command_group)
        
        # Command buttons row 1
        command_row1 = QHBoxLayout()
        
        self.hover_button = QPushButton("Hover")
        self.hover_button.clicked.connect(self.send_hover_command)
        command_row1.addWidget(self.hover_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.send_stop_command)
        command_row1.addWidget(self.stop_button)
        
        self.home_button = QPushButton("Home")
        self.home_button.clicked.connect(self.send_home_command)
        command_row1.addWidget(self.home_button)
        
        command_layout.addLayout(command_row1)
        
        # Start/Enable checkbox (only enabled in edit mode)
        self.start_checkbox = QCheckBox("Start/Enable")
        self.start_checkbox.setEnabled(False)  # Disabled by default
        self.start_checkbox.stateChanged.connect(self.toggle_start_enable)
        command_layout.addWidget(self.start_checkbox)
        
        # Send button (only enabled when not in edit mode and marker exists)
        self.send_button = QPushButton("Send Goto")
        self.send_button.setEnabled(False)
        self.send_button.clicked.connect(self.send_goto_command)
        command_layout.addWidget(self.send_button)
        
        control_layout.addWidget(command_group)
        
        # Edit Mode Controls
        edit_mode_group = QGroupBox("Edit Mode")
        edit_mode_layout = QVBoxLayout(edit_mode_group)
        
        self.edit_mode_checkbox = QCheckBox("Edit Mode")
        self.edit_mode_checkbox.stateChanged.connect(self.toggle_edit_mode)
        edit_mode_layout.addWidget(self.edit_mode_checkbox)
        
        control_layout.addWidget(edit_mode_group)
        
        # Waypoints Table
        self.waypoints_group = QGroupBox("Waypoints")
        waypoints_layout = QVBoxLayout(self.waypoints_group)
        
        self.waypoints_table = QTableWidget()
        self.waypoints_table.setColumnCount(7)
        self.waypoints_table.setHorizontalHeaderLabels([
            "No", "Position", "Orientation", "Edit", "Yaw Enable", "Landing", "Status"
        ])
        
        # Set column widths
        header = self.waypoints_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # No
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # Position
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Orientation
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Edit
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Yaw Enable
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Landing
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # Status
        
        waypoints_layout.addWidget(self.waypoints_table)
        control_layout.addWidget(self.waypoints_group)
        
        # Initially hide waypoints group
        self.waypoints_group.setVisible(False)
        
        # Enhanced marker coordinates display (for non-edit mode)
        self.marker_group = QGroupBox("Marker Parameters")
        marker_layout = QVBoxLayout(self.marker_group)
        
        # Position display
        position_layout = QHBoxLayout()
        position_layout.addWidget(QLabel("Position:"))
        self.marker_x_label = QLabel("X: 0.00")
        self.marker_y_label = QLabel("Y: 0.00")
        position_layout.addWidget(self.marker_x_label)
        position_layout.addWidget(self.marker_y_label)
        marker_layout.addLayout(position_layout)
        
        # Orientation display and edit button
        orientation_layout = QHBoxLayout()
        orientation_layout.addWidget(QLabel("Orientation:"))
        self.marker_orientation_label = QLabel("0.000 rad")
        self.edit_marker_orientation_button = QPushButton("Edit")
        self.edit_marker_orientation_button.setEnabled(False)
        self.edit_marker_orientation_button.clicked.connect(self.edit_marker_orientation)
        orientation_layout.addWidget(self.marker_orientation_label)
        orientation_layout.addWidget(self.edit_marker_orientation_button)
        marker_layout.addLayout(orientation_layout)
        
        # Yaw enable checkbox
        self.marker_yaw_checkbox = QCheckBox("Yaw Enable")
        self.marker_yaw_checkbox.setEnabled(False)
        self.marker_yaw_checkbox.stateChanged.connect(self.update_marker_yaw)
        marker_layout.addWidget(self.marker_yaw_checkbox)
        
        # Landing checkbox
        self.marker_landing_checkbox = QCheckBox("Landing")
        self.marker_landing_checkbox.setEnabled(False)
        self.marker_landing_checkbox.stateChanged.connect(self.update_marker_landing)
        marker_layout.addWidget(self.marker_landing_checkbox)
        
        control_layout.addWidget(self.marker_group)
        
        # Height Filtering Controls
        height_filter_group = QGroupBox("Height Filtering")
        height_filter_layout = QVBoxLayout(height_filter_group)
        
        max_height_layout = QHBoxLayout()
        max_height_layout.addWidget(QLabel("Max Height (m):"))
        self.max_z_spinbox = QDoubleSpinBox()
        self.max_z_spinbox.setRange(0.1, 10.0)
        self.max_z_spinbox.setSingleStep(0.1)
        self.max_z_spinbox.setValue(1.5)
        self.max_z_spinbox.setDecimals(1)
        self.max_z_spinbox.valueChanged.connect(self.update_z_filter)
        max_height_layout.addWidget(self.max_z_spinbox)
        height_filter_layout.addLayout(max_height_layout)
        
        control_layout.addWidget(height_filter_group)
        
        # View controls
        view_group = QGroupBox("View Controls")
        view_layout = QVBoxLayout(view_group)
        
        self.top_down_checkbox = QCheckBox("Lock to Top-Down View")
        self.top_down_checkbox.setChecked(True)
        self.top_down_checkbox.stateChanged.connect(self.toggle_top_down)
        view_layout.addWidget(self.top_down_checkbox)
        
        control_layout.addWidget(view_group)
        
        # File Controls
        file_group = QGroupBox("File Controls")
        file_layout = QVBoxLayout(file_group)
        
        self.save_button = QPushButton("Save Current Frame")
        self.save_button.clicked.connect(self.save_current_frame)
        file_layout.addWidget(self.save_button)
        
        self.clear_marker_button = QPushButton("Clear Marker")
        self.clear_marker_button.clicked.connect(self.clear_marker)
        file_layout.addWidget(self.clear_marker_button)
        
        control_layout.addWidget(file_group)
        control_layout.addStretch()
        layout.addWidget(control_panel, stretch=1)
        
        # Connect signals
        self.map_widget.waypoint_added.connect(self.update_waypoints_table)
        self.map_widget.waypoint_changed.connect(self.update_waypoints_table)
        self.map_widget.marker_changed.connect(self.update_marker_display)
    
    def update_tcp_status(self, connected, message):
        """Update TCP connection status."""
        if connected:
            self.tcp_label.setText(f"TCP Server: Connected")
            self.tcp_label.setStyleSheet("color: green;")
        else:
            self.tcp_label.setText(f"TCP Server: {message}")
            if "listening" in message.lower():
                self.tcp_label.setStyleSheet("color: orange;")
            else:
                self.tcp_label.setStyleSheet("color: red;")
    
    def update_websocket_status(self, connected, message):
        """Update WebSocket connection status."""
        if connected:
            self.websocket_label.setText(f"WebSocket: Connected")
            self.websocket_label.setStyleSheet("color: green;")
            self.connect_button.setText("Disconnect")
            self.connect_button.setEnabled(True)
        else:
            self.websocket_label.setText(f"WebSocket: {message}")
            self.websocket_label.setStyleSheet("color: red;")
            self.connect_button.setText("Connect")
            self.connect_button.setEnabled(True)
    
    def send_websocket_command(self, command):
        """Send command through WebSocket."""
        if self.websocket_thread and self.websocket_client.connected:
            success = self.websocket_thread.send_command_sync(command)
            if not success:
                print(f"Failed to send command: {command}")
        else:
            print(f"Cannot send command '{command}': WebSocket not connected")
    
    def send_hover_command(self):
        """Send hover command only if Start/Enable is active."""
        if self.start_checkbox.isChecked():  
            # Send hover and uncheck
            self.start_checkbox.setChecked(False)  # this will also call toggle_start_enable()
    
    def send_stop_command(self):
        """Send stop command."""
        self.send_websocket_command("stop")
    
    def send_home_command(self):
        """Send home command."""
        self.send_websocket_command("home")
    
    def toggle_start_enable(self, state):
        """Toggle start/enable state."""
        if state == Qt.Checked:
            self.start_enabled = True
            self.send_websocket_command("start")
        else:
            self.start_enabled = False
            self.send_websocket_command("hover")
    
    def send_goto_command(self):
        """Send goto command with current marker coordinates and parameters."""
        if not self.edit_mode_checkbox.isChecked():
            marker = self.map_widget.get_marker()
            if marker is not None:
                x, y = marker['position']
                orientation = marker['orientation']
                yaw_enable = 1 if marker['yaw_enable'] else 0
                landing = 1 if marker['landing'] else 0
                
                command = f"goto [{x:.2f}, {y:.2f}, {orientation:.3f}, {yaw_enable}, {landing}]"
                self.send_websocket_command(command)
                print(f"Sent goto command: {command}")
            else:
                print("No marker set for goto command")
    
    def send_coordinates_update(self):
        """Send coordinates from JSON file in edit mode."""
        if self.edit_mode_checkbox.isChecked():
            try:
                if os.path.exists(COORDINATES_FILE):
                    with open(COORDINATES_FILE, 'r') as f:
                        coordinates = json.load(f)
                    
                    # Send coordinates as JSON string
                    coordinates_json = json.dumps(coordinates)
                    self.send_websocket_command(f"coordinates {coordinates_json}")
                    
            except Exception as e:
                print(f"Error sending coordinates update: {e}")
    
    def toggle_edit_mode(self, state):
        """Toggle edit mode."""
        edit_mode = state == Qt.Checked
        self.map_widget.set_edit_mode(edit_mode)
        
        # Show/hide appropriate UI elements
        self.waypoints_group.setVisible(edit_mode)
        self.marker_group.setVisible(not edit_mode)
        
        # Enable/disable controls based on edit mode
        self.start_checkbox.setEnabled(edit_mode)
        self.send_button.setEnabled(not edit_mode)
        
        # Update table if entering edit mode
        if edit_mode:
            self.update_waypoints_table()
            self.send_coordinates_update()
        else:
            # Update marker display when exiting edit mode
            self.update_marker_display()
    
    def edit_marker_orientation(self):
        """Open joystick dialog to edit marker orientation."""
        marker = self.map_widget.get_marker()
        if marker:
            current_orientation = marker['orientation']
            
            dialog = JoystickDialog(current_orientation, self)
            if dialog.exec_() == QDialog.Accepted:
                new_orientation = dialog.orientation
                self.map_widget.update_marker(orientation=new_orientation)
                print(f"Marker orientation updated: {new_orientation:.3f} rad")
    
    def update_marker_yaw(self, state):
        """Update marker yaw enable state."""
        yaw_enable = state == Qt.Checked
        self.map_widget.update_marker(yaw_enable=yaw_enable)
        print(f"Marker yaw enable: {yaw_enable}")
    
    def update_marker_landing(self, state):
        """Update marker landing state."""
        landing = state == Qt.Checked
        self.map_widget.update_marker(landing=landing)
        print(f"Marker landing: {landing}")
    
    def update_marker_display(self):
        """Update marker parameter display."""
        marker = self.map_widget.get_marker()
        
        if marker:
            # Update position
            x, y = marker['position']
            self.marker_x_label.setText(f"X: {x:.2f}")
            self.marker_y_label.setText(f"Y: {y:.2f}")
            
            # Update orientation
            self.marker_orientation_label.setText(f"{marker['orientation']:.3f} rad")
            
            # Update checkboxes without triggering signals
            self.marker_yaw_checkbox.blockSignals(True)
            self.marker_yaw_checkbox.setChecked(marker['yaw_enable'])
            self.marker_yaw_checkbox.blockSignals(False)
            
            self.marker_landing_checkbox.blockSignals(True)
            self.marker_landing_checkbox.setChecked(marker['landing'])
            self.marker_landing_checkbox.blockSignals(False)
            
            # Enable controls
            self.edit_marker_orientation_button.setEnabled(True)
            self.marker_yaw_checkbox.setEnabled(True)
            self.marker_landing_checkbox.setEnabled(True)
            self.send_button.setEnabled(True)
        else:
            # Clear display
            self.marker_x_label.setText("X: 0.00")
            self.marker_y_label.setText("Y: 0.00")
            self.marker_orientation_label.setText("0.000 rad")
            
            self.marker_yaw_checkbox.blockSignals(True)
            self.marker_yaw_checkbox.setChecked(False)
            self.marker_yaw_checkbox.blockSignals(False)
            
            self.marker_landing_checkbox.blockSignals(True)
            self.marker_landing_checkbox.setChecked(False)
            self.marker_landing_checkbox.blockSignals(False)
            
            # Disable controls
            self.edit_marker_orientation_button.setEnabled(False)
            self.marker_yaw_checkbox.setEnabled(False)
            self.marker_landing_checkbox.setEnabled(False)
            self.send_button.setEnabled(False)
    
    def update_waypoints_table(self):
        """Update the waypoints table."""
        waypoints = self.map_widget.get_waypoints()
        self.waypoints_table.setRowCount(len(waypoints))
        
        for i, waypoint in enumerate(waypoints):
            # No column
            self.waypoints_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            
            # Position column
            pos_x, pos_y = waypoint['position']
            position_text = f"({pos_x:.2f}, {pos_y:.2f})"
            self.waypoints_table.setItem(i, 1, QTableWidgetItem(position_text))
            
            # Orientation column
            orientation_text = f"{waypoint['orientation']:.3f}"
            self.waypoints_table.setItem(i, 2, QTableWidgetItem(orientation_text))
            
            # Edit button column
            edit_button = QPushButton("Edit")
            edit_button.clicked.connect(lambda checked, idx=i: self.edit_orientation(idx))
            self.waypoints_table.setCellWidget(i, 3, edit_button)
            
            # Yaw Enable checkbox column
            yaw_checkbox = QCheckBox()
            yaw_checkbox.setChecked(waypoint['yaw_enable'])
            yaw_checkbox.stateChanged.connect(lambda state, idx=i: self.update_waypoint_yaw(idx, state))
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.addWidget(yaw_checkbox)
            layout.setAlignment(Qt.AlignCenter)
            layout.setContentsMargins(0, 0, 0, 0)
            self.waypoints_table.setCellWidget(i, 4, widget)
            
            # Landing checkbox column
            landing_checkbox = QCheckBox()
            landing_checkbox.setChecked(waypoint['landing'])
            landing_checkbox.stateChanged.connect(lambda state, idx=i: self.update_waypoint_landing(idx, state))
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.addWidget(landing_checkbox)
            layout.setAlignment(Qt.AlignCenter)
            layout.setContentsMargins(0, 0, 0, 0)
            self.waypoints_table.setCellWidget(i, 5, widget)
            
            # Status column (Add/Delete buttons)
            if waypoint['added']:
                delete_button = QPushButton("Delete")
                delete_button.clicked.connect(lambda checked, idx=i: self.delete_waypoint(idx))
                self.waypoints_table.setCellWidget(i, 6, delete_button)
            else:
                button_widget = QWidget()
                button_layout = QHBoxLayout(button_widget)
                
                add_button = QPushButton("Add")
                add_button.clicked.connect(lambda checked, idx=i: self.add_waypoint(idx))
                button_layout.addWidget(add_button)
                
                delete_button = QPushButton("Delete")
                delete_button.clicked.connect(lambda checked, idx=i: self.delete_waypoint(idx))
                button_layout.addWidget(delete_button)
                
                button_layout.setContentsMargins(0, 0, 0, 0)
                self.waypoints_table.setCellWidget(i, 6, button_widget)
    
    def edit_orientation(self, waypoint_index):
        """Open joystick dialog to edit waypoint orientation."""
        waypoints = self.map_widget.get_waypoints()
        if 0 <= waypoint_index < len(waypoints):
            current_orientation = waypoints[waypoint_index]['orientation']
            
            dialog = JoystickDialog(current_orientation, self)
            if dialog.exec_() == QDialog.Accepted:
                new_orientation = dialog.orientation
                self.map_widget.update_waypoint(waypoint_index, orientation=new_orientation)
                print(f"Waypoint {waypoint_index + 1} orientation updated: {new_orientation:.3f} rad")
        self.send_coordinates_update()
        
    def update_waypoint_yaw(self, waypoint_index, state):
        """Update waypoint yaw enable state."""
        yaw_enable = state == Qt.Checked
        self.map_widget.update_waypoint(waypoint_index, yaw_enable=yaw_enable)
        print(f"Waypoint {waypoint_index + 1} yaw enable: {yaw_enable}")
        self.send_coordinates_update()
    
    def update_waypoint_landing(self, waypoint_index, state):
        """Update waypoint landing state."""
        landing = state == Qt.Checked
        self.map_widget.update_waypoint(waypoint_index, landing=landing)
        print(f"Waypoint {waypoint_index + 1} landing: {landing}")
        self.send_coordinates_update()
    
    def add_waypoint(self, waypoint_index):
        """Add/confirm a waypoint."""
        self.map_widget.update_waypoint(waypoint_index, added=True)
        waypoints = self.map_widget.get_waypoints()
        pos = waypoints[waypoint_index]['position']
        print(f"Waypoint {waypoint_index + 1} added: ({pos[0]:.2f}, {pos[1]:.2f})")
        # Send coordinates update when waypoint is added
        self.send_coordinates_update()
    
    def delete_waypoint(self, waypoint_index):
        """Delete a waypoint."""
        self.map_widget.delete_waypoint(waypoint_index)
        print(f"Waypoint {waypoint_index + 1} deleted")
        # Send coordinates update when waypoint is deleted
        self.send_coordinates_update()
    
    def update_z_filter(self):
        """Update Z height filter values."""
        max_z = self.max_z_spinbox.value()
        self.map_widget.set_z_filter(max_z=max_z, enabled=True)
    
    def toggle_top_down(self, state):
        """Toggle top-down view mode."""
        self.map_widget.set_top_down_mode(state == Qt.Checked)
    
    def display_frame(self, points):
        """Display new complete frame."""
        self.frame_count += 1
        
        # Calculate FPS
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        if time_diff > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / time_diff)
        self.last_frame_time = current_time
        
        # Update map with complete new frame
        self.map_widget.update_current_frame(points)
    
    def update_status(self):
        """Update status display."""
        # Update marker display in non-edit mode
        if not self.edit_mode_checkbox.isChecked():
            self.update_marker_display()
        
        if self.frame_count > 0:
            self.status_label.setText(f"LIVE - Display: {self.fps:.1f} FPS")
        else:
            self.status_label.setText("Waiting for point cloud data...")
    
    def save_current_frame(self):
        """Save current raw frame (unfiltered)."""
        raw_points = self.map_widget.raw_points
        if len(raw_points) == 0:
            print("No frame to save")
            return
        
        timestamp = int(time.time())
        filename = f"pointcloud_raw_{timestamp}.ply"
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_points)
        
        # Generate colors for raw points using the full height range
        raw_heights = raw_points[:, 2]
        raw_colors = self.map_widget.generate_colors(raw_heights)
        if len(raw_colors) > 0:
            pcd.colors = o3d.utility.Vector3dVector(raw_colors)
        
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Saved raw point cloud: {filename} ({len(raw_points)} points)")
    
    def clear_marker(self):
        """Clear the marker."""
        self.map_widget.clear_marker()
        print("Marker cleared")
    
    def closeEvent(self, event):
        """Clean shutdown."""
        print("Shutting down...")
        
        # Stop TCP server
        self.tcp_receiver.stop_server()
        if self.tcp_thread:
            self.tcp_thread.quit()
            self.tcp_thread.wait()
        
        # Stop WebSocket client
        self.websocket_client.stop_client()
        if self.websocket_thread:
            self.websocket_thread.quit()
            self.websocket_thread.wait()
        
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HybridPointCloudWindow()
    window.show()
    sys.exit(app.exec_())