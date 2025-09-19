"""
drone_telemetry_handler.py

Dedicated handler for drone UDP telemetry data with thread-safe UI updates
Handles all drone data processing and UI element updates
"""

import time
import math
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtGui import QTransform, QPainter, QPixmap, QPen, QBrush
from PyQt5.QtCore import Qt, QPointF


class DroneTelemetryHandler(QObject):
    """Handles drone telemetry data processing and UI updates."""
    
    # Signals for thread-safe communication
    telemetry_updated = pyqtSignal(dict)
    connection_status_changed = pyqtSignal(bool)
    
    def __init__(self, main_window, drone_parser):
        super().__init__()
        self.main_window = main_window
        self.drone_parser = drone_parser
        self.ui = main_window.ui
        
        # Connection monitoring
        self.last_udp_packet_time = 0
        self.udp_connection_timeout = 5.0  # 5 seconds timeout
        self.flight_start_time = None
        
        # Cache original pixmaps for rotation
        self._original_top_pixmap = None
        self._original_bottom_pixmap = None
        self._original_side_pixmap = None
        
        # Setup connections
        self.setup_connections()
        self.cache_visual_assets()
        
        # Start monitoring timer
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.monitor_connection)
        self.monitor_timer.start(1000)  # Check every second
    
    def setup_connections(self):
        """Setup signal connections."""
        self.telemetry_updated.connect(self.process_telemetry_data)
        self.connection_status_changed.connect(self.update_connection_status)
    
    def cache_visual_assets(self):
        """Cache original drone visual assets."""
        try:
            from config.settings import ASSET_PATHS
            
            if 'drone_top' in ASSET_PATHS and ASSET_PATHS['drone_top'].exists():
                self._original_top_pixmap = QPixmap(str(ASSET_PATHS['drone_top']))
                
            if 'drone_bottom' in ASSET_PATHS and ASSET_PATHS['drone_bottom'].exists():
                self._original_bottom_pixmap = QPixmap(str(ASSET_PATHS['drone_bottom']))
                
            if 'drone_display' in ASSET_PATHS and ASSET_PATHS['drone_display'].exists():
                self._original_side_pixmap = QPixmap(str(ASSET_PATHS['drone_display']))
                
        except Exception as e:
            print(f"Error caching visual assets: {e}")
    
    def on_udp_packet_received(self, record):
        """Thread-safe callback for UDP packets from scapy."""
        try:
            # Update last packet time
            self.last_udp_packet_time = time.time()

            # Emit signal for UI thread processing
            self.telemetry_updated.emit(record)
            
        except Exception as e:
            print(f"Error in UDP packet callback: {e}")
    
    def process_telemetry_data(self, record):
        """Process telemetry data and update UI (runs in UI thread)."""
        try:
            if not record or 'data' not in record:
                return
            
            # Extract data using DroneParser methods
            rpy = self.drone_parser.get_rpy(record)
            position = self.drone_parser.get_position(record)
            battery = self.drone_parser.get_battery(record)
            velocity = self.drone_parser.get_velocity(record)
            
            # Update UI elements
            self.update_position_display(position)
            self.update_velocity_display(velocity)
            self.update_battery_display(battery)
            self.update_status_display(record)
            
            # Update visual drone orientation
            if rpy:
                #self.update_drone_visual_orientation(rpy)
                self.log_orientation_data(rpy)
            
            # Update point cloud with drone position
            if (position and rpy and 
                hasattr(self.main_window, 'current_view_mode') and
                self.main_window.current_view_mode == "pointcloud"):
                self.main_window.main_point_cloud.update_drone_data(
                    position['x'], position['y'], rpy['yaw']
                )
            
        except Exception as e:
            self.log_debug(f"Error processing telemetry: {e}")
    
    def update_position_display(self, position):
        """Update position-related UI elements."""
        if not position:
            return
            
        try:
            # Update position labels
            self.ui.DronePositionX.setText(f"[{position['x']:.2f}] m")
            self.ui.DronePositionY.setText(f"[{position['y']:.2f}] m") 
            self.ui.DroneHeight.setText(f"{position['z']:.2f} meter")
            
            # Update altitude slider (scale for 0-300 range)
            altitude_value = int(max(0, min(300, abs(position['z']) * 10)))
            self.ui.DroneAltitude.setValue(altitude_value)
            
        except Exception as e:
            self.log_debug(f"Error updating position: {e}")
    
    def update_velocity_display(self, velocity):
        """Update velocity-related UI elements."""
        if not velocity:
            return
            
        try:
            self.ui.DroneSpeedX.setText(f"[{velocity['xspeed']:.2f}] m/s")
            self.ui.DroneSpeedY.setText(f"[{velocity['yspeed']:.2f}] m/s")
            self.ui.DroneSpeedZ.setText(f"[{velocity['zspeed']:.2f}] m/s")
            
        except Exception as e:
            self.log_debug(f"Error updating velocity: {e}")
    
    def update_battery_display(self, battery):
        """Update battery-related UI elements."""
        if not battery or int(battery['percentage']) == 0:
            return
            
        try:
            percentage = int(battery['percentage'])
            self.ui.DroneBattery.setValue(percentage)
            
            # Dynamic color based on battery level
            if percentage < 20:
                color = "#e63946"  # Red
            elif percentage < 50:
                color = "#f77f00"  # Orange
            else:
                color = "#3ddb55"  # Green
                
            self.ui.DroneBattery.setStyleSheet(f"""
                QProgressBar {{
                    border: 2px solid #555;
                    border-radius: 10px;
                    text-align: center;
                    font-weight: bold;
                    background-color: #2b2b2b;
                    color: white;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 9px;
                }}
            """)
            
        except Exception as e:
            self.log_debug(f"Error updating battery: {e}")
    
    def update_status_display(self, record):
        """Update status-related UI elements."""
        try:
            raw_data = record.get('data', {})
            
            # Update armed/disarmed status
            if 'armed' in raw_data:
                mode_text = "ARMED" if raw_data['armed'] else "DISARMED"
                self.ui.DroneMode.setText(mode_text)
                
                # Color coding and flight timer
                if raw_data['armed']:
                    self.ui.DroneMode.setStyleSheet("color: #e63946; font-weight: bold;")
                    # Start flight timer when armed
                    if not self.flight_start_time:
                        self.flight_start_time = time.time()
                        self.log_debug("Flight timer started")
                else:
                    self.ui.DroneMode.setStyleSheet("color: #3ddb55; font-weight: bold;")
                    # Reset flight timer when disarmed
                    if self.flight_start_time:
                        self.flight_start_time = None
                        self.log_debug("Flight timer reset")
            
            # Update flight time
            self.update_flight_time_display()
            
        except Exception as e:
            self.log_debug(f"Error updating status: {e}")
    
    def update_flight_time_display(self):
        """Update flight time display."""
        try:
            if self.flight_start_time:
                flight_duration = time.time() - self.flight_start_time
                minutes = int(flight_duration // 60)
                seconds = int(flight_duration % 60)
                self.ui.DroneFlightTime.setText(f"{minutes:02d}:{seconds:02d}")
            else:
                self.ui.DroneFlightTime.setText("00:00")
                
        except Exception as e:
            self.log_debug(f"Error updating flight time: {e}")
    
    def update_drone_visual_orientation(self, rpy):
        """Update visual drone orientation based on RPY data."""
        try:
            # Extract angles in degrees
            roll_deg = rpy['roll_deg']
            pitch_deg = rpy['pitch_deg'] 
            yaw_deg = rpy['yaw_deg']
            
            # Update DroneTopView with yaw rotation
            self.rotate_drone_view(self.ui.DroneTopView, yaw_deg, '_original_top_pixmap')
            
            # Update DroneBottomView with yaw rotation  
            self.rotate_drone_view(self.ui.DroneBottomView, roll_deg, '_original_bottom_pixmap')
            
            # Update DroneSideView with pitch rotation
            self.rotate_drone_view(self.ui.DroneSideView, pitch_deg, '_original_side_pixmap')
            
            # Update compass indicators
            self.update_compass_indicators(rpy)
            
        except Exception as e:
            self.log_debug(f"Error updating visual orientation: {e}")
    
    def rotate_drone_view(self, widget, angle_deg, pixmap_attr):
        """Helper method to rotate drone view widgets."""
        try:
            if not hasattr(widget, 'setPixmap'):
                return
                
            # Get cached pixmap
            original_pixmap = getattr(self, pixmap_attr, None)
            if not original_pixmap:
                return
            
            # Create rotation transform
            transform = QTransform()
            transform.rotate(angle_deg)
            
            # Apply rotation
            rotated_pixmap = original_pixmap.transformed(transform, Qt.SmoothTransformation)
            widget.setPixmap(rotated_pixmap)
            
        except Exception as e:
            self.log_debug(f"Error rotating drone view: {e}")
    
    def update_compass_indicators(self, rpy):
        """Update compass indicators with drone orientation."""
        try:
            yaw_deg = rpy['yaw_deg']
            
            # Update compass backgrounds with heading indicator
            compass_labels = [self.ui.label_60, self.ui.label_61, self.ui.label_62]
            
            for compass_label in compass_labels:
                if compass_label:
                    self.draw_compass_with_heading(compass_label, yaw_deg)
                    
        except Exception as e:
            self.log_debug(f"Error updating compass: {e}")
    
    def draw_compass_with_heading(self, compass_label, yaw_deg):
        """Draw compass with heading indicator."""
        try:
            size = min(compass_label.width(), compass_label.height())
            if size <= 0:
                return
                
            compass_pixmap = QPixmap(size, size)
            compass_pixmap.fill(Qt.transparent)
            
            painter = QPainter(compass_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            center = QPointF(size/2, size/2)
            radius = size/2 - 10
            
            # Draw compass circle
            painter.setPen(QPen(Qt.white, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(center, radius, radius)
            
            # Draw heading indicator
            painter.setPen(QPen(Qt.red, 3))
            angle_rad = math.radians(yaw_deg - 90)  # Adjust for Qt coordinate system
            end_x = center.x() + radius * 0.8 * math.cos(angle_rad)
            end_y = center.y() + radius * 0.8 * math.sin(angle_rad)
            painter.drawLine(center, QPointF(end_x, end_y))
            
            # Draw cardinal directions
            painter.setPen(QPen(Qt.white, 1))
            painter.drawText(center.x()-5, 15, "N")
            painter.drawText(size-15, center.y()+5, "E") 
            painter.drawText(center.x()-5, size-5, "S")
            painter.drawText(5, center.y()+5, "W")
            
            painter.end()
            compass_label.setPixmap(compass_pixmap)
            
        except Exception as e:
            self.log_debug(f"Error drawing compass: {e}")
    
    def monitor_connection(self):
        """Monitor UDP connection status."""
        try:
            if not self.drone_parser or not self.drone_parser.is_running:
                self.connection_status_changed.emit(False)
                return
            
            current_time = time.time()
            if hasattr(self, 'last_udp_packet_time') and self.last_udp_packet_time > 0:
                time_since_last_packet = current_time - self.last_udp_packet_time
                connected = time_since_last_packet < self.udp_connection_timeout
                self.connection_status_changed.emit(connected)
            else:
                self.connection_status_changed.emit(False)
                
        except Exception as e:
            self.log_debug(f"Error monitoring connection: {e}")
    
    def update_connection_status(self, connected):
        """Update connection status indicators."""
        try:
            if connected:
                self.log_debug("UDP telemetry connected")
            else:
                self.log_debug("UDP telemetry disconnected")
                self.clear_drone_displays()
                
        except Exception as e:
            pass
    
    def clear_drone_displays(self):
        """Clear drone displays when connection is lost."""
        try:
            self.ui.DronePositionX.setText("[---] m")
            self.ui.DronePositionY.setText("[---] m")
            self.ui.DroneHeight.setText("--- meter")
            self.ui.DroneSpeedX.setText("[---] m/s")
            self.ui.DroneSpeedY.setText("[---] m/s")
            self.ui.DroneSpeedZ.setText("[---] m/s")
            self.ui.DroneMode.setText("DISCONNECTED")
            self.ui.DroneMode.setStyleSheet("color: #888888; font-weight: bold;")
            self.ui.DroneFlightTime.setText("00:00")
            
            # Reset flight timer
            self.flight_start_time = None
            
        except Exception as e:
            pass
    
    def log_orientation_data(self, rpy):
        """Log orientation data for debugging."""
        try:
            # print(f"Orientation - Roll: {rpy['roll_deg']:.1f}°, "
            #               f"Pitch: {rpy['pitch_deg']:.1f}°, Yaw: {rpy['yaw_deg']:.1f}°")
            pass
        except Exception as e:
            pass
    
    def log_debug(self, message):
        """Log debug message through main window."""
        if hasattr(self.main_window, 'log_debug'):
            self.main_window.log_debug(message)
        else:
            print(f"[TelemetryHandler] {message}")
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.monitor_timer:
                self.monitor_timer.stop()
                
            # Save telemetry data if available
            if self.drone_parser and len(self.drone_parser) > 0:
                filename = f"drone_udp_telemetry_{int(time.time())}.json"
                saved_file = self.drone_parser.save_data(filename)
                print(f"Saved {len(self.drone_parser)} UDP telemetry records to {saved_file}")
                
        except Exception as e:
            print(f"Error during telemetry handler cleanup: {e}")