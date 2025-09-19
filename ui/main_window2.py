"""
main_window.py (FIXED STARTUP SEQUENCE)

Drone Control Center dengan startup sequence yang benar untuk DroneParser
"""

import sys
import time
import threading
from pathlib import Path
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel
from PyQt5.QtCore import QTimer, Qt

# Import UI yang sudah dibuat
from activity_ui import Ui_MainWindow

# Import komponen fungsional
from config.settings import (
    APP_CONFIG, UI_CONFIG, ASSET_PATHS, NETWORK_CONFIG, FILE_PATHS
)
from .widgets.point_cloud_widget import SmoothPointCloudWidget
from .widgets.video_stream_widget import VideoStreamWidget, RTSPStreamWidget, TCPVideoStreamWidget
from .widgets.joystick_dialog import JoystickDialog
from core.tcp_receiver import TCPDataReceiver, TCPServerThread
from core.websocket_client import WebSocketCommandClient, WebSocketCommandThread
from core.drone_parser import DroneParser

# Import telemetry handler
from .drone_telemetry_handler import DroneTelemetryHandler


class DroneControlMainWindow(QMainWindow):
    """Main Window yang mengintegrasikan UI design dengan fungsionalitas lengkap."""
    
    def __init__(self):
        super().__init__()
        
        # Setup UI dari design
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Initialize data storage
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0.0
        self.start_enabled = False
        
        # View mode tracking
        self.current_view_mode = "pointcloud"  # "pointcloud" or "video"
        
        # Initialize components step by step
        self.setup_functional_widgets()
        self.update_asset_paths()
        self.setup_communication()
        self.connect_signals()
        self.setup_timers()
        
        # Start services after UI is ready
        QTimer.singleShot(100, self.start_services)
        
        # Log startup completion
        QTimer.singleShot(500, lambda: self.log_debug("Drone Control Center started successfully"))
    
    def setup_functional_widgets(self):
        """Replace placeholder labels dengan functional widgets."""
        
        # Replace SwitchView_1 dengan main point cloud widget
        self.main_point_cloud = SmoothPointCloudWidget()
        self.main_point_cloud.setParent(self.ui.centralwidget)
        self.main_point_cloud.setGeometry(self.ui.SwitchView_1.geometry())
        self.main_point_cloud.setStyleSheet(self.ui.SwitchView_1.styleSheet())
        self.ui.SwitchView_1.hide()
        
        # Replace SwitchView_2 dengan video streaming widget  
        self.video_stream = VideoStreamWidget()
        self.video_stream.setParent(self.ui.frame_8)
        self.video_stream.setGeometry(self.ui.SwitchView_2.geometry())
        self.video_stream.setStyleSheet(self.ui.SwitchView_2.styleSheet())
        self.ui.SwitchView_2.hide()
        
        # Initially show point cloud and hide video
        self.main_point_cloud.show()
        self.video_stream.hide()
        
        # Setup current marker untuk single command mode
        self.current_marker = None
        self.edit_mode = False
        
        self.log_debug("Functional widgets initialized")
    
    def update_asset_paths(self):
        """Update asset paths untuk styling yang menggunakan relative paths."""
        widgets_with_assets = [
            (self.ui.label_62, 'compass'),
            (self.ui.label_60, 'compass'), 
            (self.ui.label_61, 'compass'),
            (self.ui.DroneBottomView, 'drone_bottom'),
            (self.ui.DroneTopView, 'drone_top'),
            (self.ui.DroneSideView, 'drone_display'),
            (self.ui.label, 'logo'),
            (self.ui.label_2, 'drone_display'),
            (self.ui.label_9, 'logo'),
            (self.ui.btAutonomousEmergency, 'emergency'),
            (self.ui.DroneAltitude, 'altitude')
        ]
        
        for widget, asset_key in widgets_with_assets:
            if asset_key in ASSET_PATHS and ASSET_PATHS[asset_key].exists():
                current_style = widget.styleSheet()
                asset_path = str(ASSET_PATHS[asset_key]).replace('\\', '/')
                if 'image: url(' in current_style:
                    import re
                    new_style = re.sub(
                        r'image: url\(["\']?[^"\']*["\']?\);',
                        f'image: url("{asset_path}");',
                        current_style
                    )
                    widget.setStyleSheet(new_style)
        
        self.log_debug("Asset paths updated")
    
    def setup_communication(self):
        """Initialize communication components with proper startup sequence."""
        
        # Initialize communication components
        self.tcp_receiver = TCPDataReceiver()
        self.tcp_thread = None
        self.websocket_client = WebSocketCommandClient()
        self.websocket_thread = None
        
        # Initialize drone parser and telemetry handler
        self.drone_parser = None
        self.telemetry_handler = None
        
        try:
            self.log_debug("Initializing DroneParser for UDP telemetry...")
            
            drone_port = NETWORK_CONFIG.get('drone_data_port', 8889)
            
            # Initialize DroneParser with proper error handling
            try:
                self.drone_parser = DroneParser(port=drone_port, max_records=1000)
                self.log_debug(f"DroneParser initialized with port {drone_port}")
            except TypeError as e:
                if "max_records" in str(e):
                    self.log_debug("Trying DroneParser with port only...")
                    self.drone_parser = DroneParser(port=drone_port)
                else:
                    self.log_debug("Trying DroneParser with default parameters...")
                    self.drone_parser = DroneParser()
            
            # Create telemetry handler BEFORE starting parser
            if self.drone_parser:
                self.telemetry_handler = DroneTelemetryHandler(self, self.drone_parser)
                self.log_debug("Telemetry handler created successfully")
            else:
                self.log_debug("Failed to create DroneParser - telemetry disabled")
                
        except ImportError as e:
            self.log_debug(f"DroneParser import failed: {e}")
            self.log_debug("Make sure scapy is installed: pip install scapy")
            self.drone_parser = None
        except Exception as e:
            self.log_debug(f"Failed to initialize drone parser: {e}")
            self.drone_parser = None
    
    def start_drone_parser(self):
        """Start drone parser with telemetry handler callback."""
        if self.drone_parser and self.telemetry_handler:
            try:
                # Start UDP packet capture with telemetry handler callback
                self.drone_parser.start(callback=self.telemetry_handler.on_udp_packet_received)
                self.log_debug(f"DroneParser started - listening for UDP packets on port {self.drone_parser.port}")
                return True
            except Exception as e:
                self.log_debug(f"Failed to start DroneParser: {e}")
                return False
        else:
            self.log_debug("DroneParser or TelemetryHandler not available")
            return False
    
    def connect_signals(self):
        """Connect all signals dan event handlers."""
        
        # TCP signals
        self.tcp_receiver.frame_received.connect(self.display_frame)
        self.tcp_receiver.connection_status.connect(self.update_tcp_status)
        
        # WebSocket signals
        self.websocket_client.connection_status.connect(self.update_websocket_status)
        
        # Point cloud widget signals
        self.main_point_cloud.waypoint_added.connect(self.update_waypoints_table)
        self.main_point_cloud.waypoint_changed.connect(self.update_waypoints_table)
        self.main_point_cloud.marker_changed.connect(self.update_marker_display)
        
        # Video stream signals
        self.video_stream.frame_received.connect(self.update_video_status)
        
        # Connect UI elements ke functions
        self.ui.CommandConnect.clicked.connect(self.toggle_connection)
        self.ui.btAutonomousEmergency.clicked.connect(self.emergency_stop)
        self.ui.DroneSwitch.clicked.connect(self.switch_views)
        
        # Single Command controls
        self.ui.scHover.clicked.connect(lambda: self.send_websocket_command("hover"))
        self.ui.scSendGoto.clicked.connect(self.send_goto_command)
        self.ui.scEdit.clicked.connect(self.edit_marker_orientation)
        self.ui.scClearMarker.clicked.connect(self.clear_marker)
        self.ui.scYawEnable.stateChanged.connect(self.update_marker_yaw)
        self.ui.scLanding.stateChanged.connect(self.update_marker_landing)
        
        # Multiple Command controls
        self.ui.mcEditMode.stateChanged.connect(self.toggle_edit_mode)
        self.ui.mcViewControls.stateChanged.connect(self.update_view_controls)
        self.ui.mcHeightFiltering.valueChanged.connect(self.update_height_filter)
        self.ui.mcHover.clicked.connect(lambda: self.send_websocket_command("hover"))
        self.ui.mcHome.clicked.connect(lambda: self.send_websocket_command("home"))
        self.ui.mcSaveMaps.clicked.connect(self.save_current_frame)
        self.ui.mcSendCommand.clicked.connect(self.send_multiple_commands)
        self.ui.mcDialOrientation.valueChanged.connect(self.on_orientation_dial_changed)
        
        # Altitude slider
        self.ui.DroneAltitude.valueChanged.connect(self.update_altitude_display)
        
        self.log_debug("Signals connected")
    
    def setup_timers(self):
        """Setup update timers."""
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(UI_CONFIG['update_intervals']['status_update'])
        
        self.log_debug("Timers initialized")
    
    def start_services(self):
        """Start communication services in correct order."""
        self.log_debug("Starting services...")
        
        # Start TCP server first
        self.start_tcp_server()
        
        # Start drone parser - THIS IS THE KEY PART
        if self.start_drone_parser():
            self.log_debug("All services started successfully")
        else:
            self.log_debug("Services started (UDP telemetry unavailable)")
        
        # Initialize clear displays
        self.clear_drone_displays_initial()
    
    def start_tcp_server(self):
        """Start TCP server thread."""
        if self.tcp_thread is None or not self.tcp_thread.isRunning():
            self.tcp_thread = TCPServerThread(self.tcp_receiver)
            self.tcp_thread.start()
            self.log_debug("TCP server started")
    
    def clear_drone_displays_initial(self):
        """Set initial clear state for drone displays."""
        try:
            self.ui.DronePositionX.setText("[0.00] m")
            self.ui.DronePositionY.setText("[0.00] m")
            self.ui.DroneHeight.setText("0.00 meter")
            self.ui.DroneSpeedX.setText("[0.000] m/s")
            self.ui.DroneSpeedY.setText("[0.000] m/s")
            self.ui.DroneSpeedZ.setText("[0.000] m/s")
            self.ui.DroneMode.setText("WAITING")
            self.ui.DroneMode.setStyleSheet("color: #888888; font-weight: bold;")
            self.ui.DroneFlightTime.setText("00:00")
            self.ui.DroneBattery.setValue(0)
        except Exception as e:
            self.log_debug(f"Error setting initial display state: {e}")
    
    # All the rest of the methods remain the same...
    # (WebSocket methods, view management, waypoint management, etc.)
    
    def toggle_connection(self):
        """Toggle WebSocket connection."""
        if self.websocket_client.connected:
            self.disconnect_websocket()
        else:
            self.connect_websocket()
    
    def connect_websocket(self):
        """Connect to WebSocket server."""
        if self.websocket_thread is None or not self.websocket_thread.isRunning():
            self.websocket_client.start_client()
            self.websocket_thread = WebSocketCommandThread(self.websocket_client)
            self.websocket_thread.start()
    
    def disconnect_websocket(self):
        """Disconnect from WebSocket server."""
        self.websocket_client.stop_client()
        if self.websocket_thread:
            self.websocket_thread.quit()
            self.websocket_thread.wait()
            self.websocket_thread = None
    
    def send_websocket_command(self, command):
        """Send command through WebSocket."""
        if self.websocket_thread and self.websocket_client.connected:
            success = self.websocket_thread.send_command_sync(command)
            if not success:
                self.log_debug(f"Failed to send: {command}")
                return False
            else:
                return True
        else:
            self.log_debug(f"Cannot send '{command}': Not connected")
            return True
    
    def emergency_stop(self):
        """Emergency stop command."""
        self.send_websocket_command("stop")
        self.log_debug("EMERGENCY STOP ACTIVATED")
    
    def switch_views(self):
        """Switch/swap positions between point cloud and video stream widgets."""
        # Get current geometries dan parents
        pc_geometry = self.main_point_cloud.geometry()
        pc_parent = self.main_point_cloud.parent()
        pc_style = self.main_point_cloud.styleSheet()
        
        video_geometry = self.video_stream.geometry()
        video_parent = self.video_stream.parent()
        video_style = self.video_stream.styleSheet()
        
        # Swap positions
        self.main_point_cloud.setParent(video_parent)
        self.main_point_cloud.setGeometry(video_geometry)
        self.main_point_cloud.setStyleSheet(video_style)
        
        self.video_stream.setParent(pc_parent)
        self.video_stream.setGeometry(pc_geometry)
        self.video_stream.setStyleSheet(pc_style)
        
        # Show both widgets setelah swap
        self.main_point_cloud.show()
        self.video_stream.show()
        
        # Toggle main view mode untuk waypoint controls
        if self.current_view_mode == "pointcloud":
            self.current_view_mode = "video"
            self.ui.DroneSwitch.setText("Point Cloud Main")
            
            # Disable waypoint controls
            self.ui.mcEditMode.setEnabled(False)
            self.ui.scEdit.setEnabled(False)
            self.ui.scSendGoto.setEnabled(False)
            self.ui.scClearMarker.setEnabled(False)
            
            self.log_debug("Switched: Video main, Point cloud secondary")
        else:
            self.current_view_mode = "pointcloud"
            self.ui.DroneSwitch.setText("Video Main")
            
            # Re-enable waypoint controls
            self.ui.mcEditMode.setEnabled(True)
            self.update_marker_display()
            
            self.log_debug("Switched: Point cloud main, Video secondary")
    
    def display_frame(self, points):
        """Display new point cloud frame."""
        self.frame_count += 1
        
        # Calculate FPS
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        if time_diff > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / time_diff)
        self.last_frame_time = current_time
        
        # Update point cloud widget
        self.main_point_cloud.update_current_frame(points)
    
    def update_video_status(self):
        """Update video stream status."""
        video_info = self.video_stream.get_video_info()
        pass
    
    def update_status(self):
        """Update status display."""
        if self.current_view_mode == "pointcloud":
            if self.frame_count > 0:
                status_text = f"LIVE - Point Cloud: {self.fps:.1f} FPS"
            else:
                status_text = "Waiting for point cloud data..."
        else:
            video_info = self.video_stream.get_video_info()
            if video_info['connected']:
                status_text = f"LIVE - Video: {video_info['fps']:.1f} FPS"
            else:
                status_text = "Waiting for video stream..."
        
        # Update command control status based on telemetry
        if self.telemetry_handler and hasattr(self.telemetry_handler, 'last_udp_packet_time'):
            if self.telemetry_handler.last_udp_packet_time > 0:
                time_since_packet = time.time() - self.telemetry_handler.last_udp_packet_time
                if time_since_packet < 5.0:  # 5 second timeout
                    self.ui.CommandControl.setText("Active")
                else:
                    self.ui.CommandControl.setText("Idle")
            else:
                self.ui.CommandControl.setText("Waiting")
        else:
            self.ui.CommandControl.setText("Waiting")
    
    def update_tcp_status(self, connected, message):
        """Update TCP connection status."""
        if connected:
            self.ui.CommandOnline.setStyleSheet("""
                background-color: rgb(0, 255, 0);
                border: 0px;
                border-radius: 10px;
            """)
            self.ui.CommandStatus.setText("Connected")
            self.ui.btQrcodeOnline_3.setStyleSheet("""
                background-color: rgb(0, 255, 0);
                border: 0px;
            """)
        else:
            self.ui.CommandOnline.setStyleSheet("""
                background-color: rgb(255, 0, 0);
                border: 0px;
                border-radius: 10px;
            """)
            self.ui.CommandStatus.setText("Disconnected")
            self.ui.btQrcodeOnline_3.setStyleSheet("""
                background-color: rgb(255, 0, 0);
                border: 0px;
            """)
    
    def update_websocket_status(self, connected, message):
        """Update WebSocket connection status."""
        if connected:
            self.ui.CommandConnect.setText("DISCONNECT")
            self.ui.CommandMode.setText("Connected")
        else:
            self.ui.CommandConnect.setText("CONNECT")
            self.ui.CommandMode.setText("Disconnected")
    
    # Waypoint and marker management methods
    def toggle_edit_mode(self, state):
        """Toggle edit mode for waypoint management."""
        if self.current_view_mode != "pointcloud":
            self.ui.mcEditMode.setChecked(False)
            return
            
        self.edit_mode = state == Qt.Checked
        self.main_point_cloud.set_edit_mode(self.edit_mode)
        
        if self.edit_mode:
            self.update_waypoints_table()
            self.log_debug("Edit mode enabled")
        else:
            self.log_debug("Edit mode disabled")
    
    def update_view_controls(self, state):
        """Update view controls."""
        if self.current_view_mode == "pointcloud":
            top_down_enabled = state == Qt.Checked
            self.main_point_cloud.set_top_down_mode(top_down_enabled)
            self.log_debug(f"Top-down view: {'Locked' if top_down_enabled else 'Unlocked'}")
    
    def update_height_filter(self, value):
        """Update height filtering."""
        if self.current_view_mode == "pointcloud":
            self.main_point_cloud.set_z_filter(max_z=value, enabled=True)
            self.log_debug(f"Height filter: {value:.1f}m")
    
    # Add all other methods here (waypoint management, etc.)
    # ... (keeping the rest of the methods from the previous version)
    
    def log_debug(self, message):
        """Log message ke debugging console."""
        try:
            timestamp = time.strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}"
            
            self.ui.tbDebugging.append(formatted_message)
            
            # Keep only last 100 lines
            document = self.ui.tbDebugging.document()
            if document.blockCount() > 100:
                cursor = self.ui.tbDebugging.textCursor()
                cursor.movePosition(cursor.Start)
                cursor.select(cursor.LineUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar()
        except Exception as e:
            print(f"[DEBUG] {message}")
    
    def closeEvent(self, event):
        """Clean shutdown with proper service stopping."""
        print("Shutting down Drone Control Center...")
        
        # Stop drone parser first
        if hasattr(self, 'drone_parser') and self.drone_parser:
            try:
                print("Stopping DroneParser...")
                self.drone_parser.stop()
                print("DroneParser stopped")
            except Exception as e:
                print(f"Error stopping DroneParser: {e}")
        
        # Cleanup telemetry handler
        if hasattr(self, 'telemetry_handler') and self.telemetry_handler:
            try:
                self.telemetry_handler.cleanup()
                print("Telemetry handler cleaned up")
            except Exception as e:
                print(f"Error cleaning telemetry handler: {e}")
        
        # Stop other services
        if hasattr(self, 'tcp_receiver'):
            self.tcp_receiver.stop_server()
        if hasattr(self, 'tcp_thread') and self.tcp_thread:
            self.tcp_thread.quit()
            self.tcp_thread.wait()
        
        if hasattr(self, 'websocket_client'):
            self.websocket_client.stop_client()
        if hasattr(self, 'websocket_thread') and self.websocket_thread:
            self.websocket_thread.quit()
            self.websocket_thread.wait()
        
        # Save waypoints
        if hasattr(self, 'main_point_cloud'):
            try:
                self.main_point_cloud.save_waypoints_to_json()
                print("Waypoints saved")
            except Exception as e:
                print(f"Error saving waypoints: {e}")
        
        print("Shutdown complete")
        event.accept()