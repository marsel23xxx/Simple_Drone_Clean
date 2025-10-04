"""
main_window.py (CLEANED VERSION)

Drone Control Center Main Window dengan integrasi telemetry yang bersih
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
from ..video_stream_widget import VideoStreamWidget, RTSPStreamWidget, TCPVideoStreamWidget
from .widgets.joystick_dialog import JoystickDialog
from core.tcp_receiver import TCPDataReceiver, TCPServerThread
from core.websocket_client import WebSocketCommandClient, WebSocketCommandThread
from core.drone_parser import DroneParser

# Import telemetry handler
from .widgets.drone_telemetry_handler import DroneTelemetryHandler
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
        
        # Initialize communication components
        self.setup_communication()
        
        # Replace placeholder labels dengan functional widgets
        self.setup_functional_widgets()
        
        # Connect signals dan setup event handlers
        self.connect_signals()
        
        # Start services
        self.start_services()
        
        # Setup timers
        self.setup_timers()
        
        # Update asset paths untuk menggunakan path yang benar
        self.update_asset_paths()
    
    def setup_communication(self):
        """Initialize communication components with UDP telemetry handler."""
        # TCP server for point cloud data
        self.tcp_receiver = TCPDataReceiver()
        self.tcp_thread = None
        
        # WebSocket client for commands
        self.websocket_client = WebSocketCommandClient()
        self.websocket_thread = None
        
        # Initialize drone parser for UDP telemetry
        self.drone_parser = None
        self.telemetry_handler = None
        
        try:
            print("Initializing DroneParser for UDP telemetry capture...")
            
            drone_port = 8889
            
            # Initialize DroneParser
            try:
                self.drone_parser = DroneParser(port=drone_port, max_records=1000)
                self.telemetry_handler = DroneTelemetryHandler(self, self.drone_parser)
        
                # Start UDP packet capture with telemetry handler callback
                self.drone_parser.start(callback=self.telemetry_handler.on_udp_packet_received)
                print(f"DroneParser started - capturing UDP packets on port {drone_port}")
            except TypeError:
                try:
                    self.drone_parser = DroneParser(port=drone_port)
                    self.telemetry_handler = DroneTelemetryHandler(self, self.drone_parser)
        
                    # Start UDP packet capture with telemetry handler callback
                    self.drone_parser.start(callback=self.telemetry_handler.on_udp_packet_received)
                    print(f"DroneParser started - capturing UDP packets on port {drone_port}")
                except:
                    self.drone_parser = DroneParser()
        
        except ImportError as e:
            print(f"DroneParser import failed: {e}")
            print("Make sure scapy is installed: pip install scapy")
            self.drone_parser = None
        except Exception as e:
            print(f"Failed to initialize UDP drone parser: {e}")
            self.drone_parser = None
    
    def setup_functional_widgets(self):
        """Replace placeholder labels dengan functional widgets."""
        
        # Replace SwitchView_1 dengan main point cloud widget
        self.main_point_cloud = SmoothPointCloudWidget()
        self.main_point_cloud.setParent(self.ui.centralwidget)
        self.main_point_cloud.setGeometry(self.ui.SwitchView_1.geometry())
        self.main_point_cloud.setStyleSheet(self.ui.SwitchView_1.styleSheet())
        self.ui.SwitchView_1.hide()  # Hide original label
        
        # Replace SwitchView_2 dengan video streaming widget  
        self.video_stream = VideoStreamWidget()
        self.video_stream.setParent(self.ui.frame_8)
        self.video_stream.setGeometry(self.ui.SwitchView_2.geometry())
        self.video_stream.setStyleSheet(self.ui.SwitchView_2.styleSheet())
        self.ui.SwitchView_2.hide()  # Hide original label
        
        # Initially show point cloud and hide video
        self.main_point_cloud.show()
        self.video_stream.hide()
        
        # Setup current marker untuk single command mode
        self.current_marker = None
        
        # Setup edit mode tracking
        self.edit_mode = False
    
    def update_asset_paths(self):
        """Update asset paths untuk styling yang menggunakan relative paths."""
        # Update paths in stylesheets to use absolute paths
        widgets_with_assets = [
            (self.ui.label_62, 'compass'),
            (self.ui.label_60, 'compass'), 
            (self.ui.label_61, 'compass'),
            # (self.ui.DroneBottomView, 'drone_bottom'),
            # (self.ui.DroneTopView, 'drone_top'),
            # (self.ui.DroneSideView, 'drone_display'),
            (self.ui.label, 'logo'),
            (self.ui.label_2, 'drone_display'),
            (self.ui.label_9, 'logo'),
            (self.ui.btAutonomousEmergency, 'emergency'),
            (self.ui.DroneAltitude, 'altitude'),
            (self.ui.DroneHeight, 'height')
        ]
        
        for widget, asset_key in widgets_with_assets:
            if asset_key in ASSET_PATHS and ASSET_PATHS[asset_key].exists():
                current_style = widget.styleSheet()
                # Replace relative path dengan absolute path
                asset_path = str(ASSET_PATHS[asset_key]).replace('\\', '/')
                if 'image: url(' in current_style:
                    # Update existing image URL
                    import re
                    new_style = re.sub(
                        r'image: url\(["\']?[^"\']*["\']?\);',
                        f'image: url("{asset_path}");',
                        current_style
                    )
                    widget.setStyleSheet(new_style)
    
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
    
    def setup_timers(self):
        """Setup update timers."""
        # Status update timer
        self.status_timer = QTimer()
        #self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(UI_CONFIG['update_intervals']['status_update'])
    
    def start_services(self):
        """Start communication services."""
        self.start_tcp_server()
        
        # Initialize video streaming (example with RTSP)
        # Uncomment dan sesuaikan dengan sumber video Anda
        # self.start_video_stream()
        
    def start_tcp_server(self):
        """Start TCP server thread."""
        if self.tcp_thread is None or not self.tcp_thread.isRunning():
            self.tcp_thread = TCPServerThread(self.tcp_receiver)
            self.tcp_thread.start()
    
    def start_video_stream(self, stream_url=None):
        """Start video streaming."""
        # Example: Start RTSP stream
        if stream_url and stream_url.startswith('rtsp://'):
            # Replace current video widget dengan RTSP widget
            old_geometry = self.video_stream.geometry()
            old_parent = self.video_stream.parent()
            old_style = self.video_stream.styleSheet()
            
            self.video_stream.hide()
            self.video_stream = RTSPStreamWidget(stream_url)
            self.video_stream.setParent(old_parent)
            self.video_stream.setGeometry(old_geometry)
            self.video_stream.setStyleSheet(old_style)
            self.video_stream.frame_received.connect(self.update_video_status)
            
            # Start the stream
            success = self.video_stream.start_stream()
            if success:
                self.log_debug(f"Video stream started: {stream_url}")
            else:
                self.log_debug(f"Failed to start video stream: {stream_url}")
        else:
            # Setup TCP video streaming
            if hasattr(self.tcp_receiver, 'video_frame_received'):
                self.video_stream.set_tcp_receiver(self.tcp_receiver)
                self.log_debug("TCP video stream configured")
    
    # WebSocket methods
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
                print(f"Failed to send command: {command}")
                self.log_debug(f"Failed to send: {command}")
                return False
            else:
                return True
        else:
            print(f"Cannot send command '{command}': WebSocket not connected")
            self.log_debug(f"Cannot send '{command}': Not connected")
            return True
    
    def emergency_stop(self):
        """Emergency stop command."""
        self.send_websocket_command("stop")
        self.log_debug("EMERGENCY STOP ACTIVATED")
        print("EMERGENCY STOP ACTIVATED")
    
    # View and display methods
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
        # Could update UI with video stream info
        pass
    
    
    
    # Connection status methods
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
    
    # Waypoint and marker management
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
    
    def update_waypoints_table(self):
        """Update waypoints display table."""
        if self.current_view_mode != "pointcloud":
            return
            
        waypoints = self.main_point_cloud.get_waypoints()
        self.ui.mcDisplayData.setRowCount(len(waypoints))
        
        for i, waypoint in enumerate(waypoints):
            pos_x, pos_y = waypoint['position']
            
            from PyQt5.QtWidgets import QTableWidgetItem, QPushButton, QCheckBox
            self.ui.mcDisplayData.setItem(i, 0, 
                QTableWidgetItem(f"({pos_x:.2f}, {pos_y:.2f})"))
            
            self.ui.mcDisplayData.setItem(i, 1, 
                QTableWidgetItem(f"{waypoint['orientation']:.3f}"))
            
            # Edit button
            edit_btn = QPushButton("Edit")
            edit_btn.clicked.connect(lambda checked, idx=i: self.edit_waypoint_orientation(idx))
            self.ui.mcDisplayData.setCellWidget(i, 2, edit_btn)
            
            # Yaw Enable
            yaw_check = QCheckBox()
            yaw_check.setChecked(waypoint['yaw_enable'])
            yaw_check.stateChanged.connect(lambda state, idx=i: self.update_waypoint_yaw(idx, state))
            self.ui.mcDisplayData.setCellWidget(i, 3, yaw_check)
            
            # Landing
            land_check = QCheckBox()
            land_check.setChecked(waypoint['landing'])
            land_check.stateChanged.connect(lambda state, idx=i: self.update_waypoint_landing(idx, state))
            self.ui.mcDisplayData.setCellWidget(i, 4, land_check)
            
            # Action
            if waypoint['added']:
                action_btn = QPushButton("Delete")
                action_btn.clicked.connect(lambda checked, idx=i: self.delete_waypoint(idx))
            else:
                action_btn = QPushButton("Add")
                action_btn.clicked.connect(lambda checked, idx=i: self.add_waypoint(idx))
            self.ui.mcDisplayData.setCellWidget(i, 5, action_btn)
    
    def update_marker_display(self):
        """Update marker display untuk single command mode."""
        if self.current_view_mode != "pointcloud":
            # Clear display and disable controls
            self.ui.scPositionX.setText("0.00")
            self.ui.scPositionY.setText("0.00")
            self.ui.scOrientation.setText("0.0000 rad")
            self.ui.scYawEnable.setChecked(False)
            self.ui.scLanding.setChecked(False)
            self.ui.scEdit.setEnabled(False)
            self.ui.scSendGoto.setEnabled(False)
            self.ui.scClearMarker.setEnabled(False)
            self.current_marker = None
            return
        
        marker = self.main_point_cloud.get_marker()
        
        if marker:
            x, y = marker['position']
            self.ui.scPositionX.setText(f"{x:.2f}")
            self.ui.scPositionY.setText(f"{y:.2f}")
            self.ui.scOrientation.setText(f"{marker['orientation']:.4f} rad")
            
            self.ui.scYawEnable.blockSignals(True)
            self.ui.scYawEnable.setChecked(marker['yaw_enable'])
            self.ui.scYawEnable.blockSignals(False)
            
            self.ui.scLanding.blockSignals(True)
            self.ui.scLanding.setChecked(marker['landing'])
            self.ui.scLanding.blockSignals(False)
            
            # Enable controls
            self.ui.scEdit.setEnabled(True)
            self.ui.scSendGoto.setEnabled(True)
            self.ui.scClearMarker.setEnabled(True)
            
            self.current_marker = marker
        else:
            # Clear display
            self.ui.scPositionX.setText("0.00")
            self.ui.scPositionY.setText("0.00")
            self.ui.scOrientation.setText("0.0000 rad")
            
            self.ui.scYawEnable.blockSignals(True)
            self.ui.scYawEnable.setChecked(False)
            self.ui.scYawEnable.blockSignals(False)
            
            self.ui.scLanding.blockSignals(True)
            self.ui.scLanding.setChecked(False)
            self.ui.scLanding.blockSignals(False)
            
            # Disable controls
            self.ui.scEdit.setEnabled(False)
            self.ui.scSendGoto.setEnabled(False)
            self.ui.scClearMarker.setEnabled(False)
            
            self.current_marker = None
    
    # Marker and waypoint actions
    def edit_marker_orientation(self):
        """Edit marker orientation menggunakan joystick dialog."""
        if self.current_marker and self.current_view_mode == "pointcloud":
            current_orientation = self.current_marker['orientation']
            
            dialog = JoystickDialog(current_orientation, self)
            if dialog.exec_() == dialog.Accepted:
                new_orientation = dialog.orientation
                self.main_point_cloud.update_marker(orientation=new_orientation)
                self.log_debug(f"Marker orientation updated: {new_orientation:.3f} rad")
    
    def clear_marker(self):
        """Clear current marker."""
        if self.current_view_mode == "pointcloud":
            self.main_point_cloud.clear_marker()
            self.log_debug("Marker cleared")
    
    def send_goto_command(self):
        """Send goto command dengan current marker data."""
        if self.current_marker and self.current_view_mode == "pointcloud":
            x, y = self.current_marker['position']
            orientation = self.current_marker['orientation']
            yaw_enable = 1 if self.current_marker['yaw_enable'] else 0
            landing = 1 if self.current_marker['landing'] else 0
            
            command = f"goto [{x:.2f}, {y:.2f}, {orientation:.3f}, {yaw_enable}, {landing}]"
            self.send_websocket_command(command)
            self.log_debug(f"Sent goto: {command}")
    
    def update_marker_yaw(self, state):
        """Update marker yaw enable state."""
        if self.current_marker and self.current_view_mode == "pointcloud":
            yaw_enable = state == Qt.Checked
            self.main_point_cloud.update_marker(yaw_enable=yaw_enable)
            self.log_debug(f"Marker yaw enable: {yaw_enable}")
    
    def update_marker_landing(self, state):
        """Update marker landing state."""
        if self.current_marker and self.current_view_mode == "pointcloud":
            landing = state == Qt.Checked
            self.main_point_cloud.update_marker(landing=landing)
            self.log_debug(f"Marker landing: {landing}")
    
    def edit_waypoint_orientation(self, index):
        """Edit waypoint orientation."""
        if self.current_view_mode != "pointcloud":
            return
            
        waypoints = self.main_point_cloud.get_waypoints()
        if 0 <= index < len(waypoints):
            current_orientation = waypoints[index]['orientation']
            
            dialog = JoystickDialog(current_orientation, self)
            if dialog.exec_() == dialog.Accepted:
                new_orientation = dialog.orientation
                self.main_point_cloud.update_waypoint(index, orientation=new_orientation)
                self.log_debug(f"Waypoint {index + 1} orientation: {new_orientation:.3f} rad")
    
    def update_waypoint_yaw(self, index, state):
        """Update waypoint yaw enable state."""
        if self.current_view_mode == "pointcloud":
            yaw_enable = state == Qt.Checked
            self.main_point_cloud.update_waypoint(index, yaw_enable=yaw_enable)
            self.log_debug(f"Waypoint {index + 1} yaw enable: {yaw_enable}")
    
    def update_waypoint_landing(self, index, state):
        """Update waypoint landing state."""
        if self.current_view_mode == "pointcloud":
            landing = state == Qt.Checked
            self.main_point_cloud.update_waypoint(index, landing=landing)
            self.log_debug(f"Waypoint {index + 1} landing: {landing}")
    
    def add_waypoint(self, index):
        """Add waypoint dan kirim ke server."""
        if self.current_view_mode == "pointcloud":
            self.main_point_cloud.update_waypoint(index, added=True)
            waypoints = self.main_point_cloud.get_waypoints()
            pos = waypoints[index]['position']
            self.log_debug(f"Waypoint {index + 1} added: ({pos[0]:.2f}, {pos[1]:.2f})")
            self.send_waypoints_to_server()
    
    def delete_waypoint(self, index):
        """Delete waypoint dan kirim ke server."""
        if self.current_view_mode == "pointcloud":
            self.main_point_cloud.delete_waypoint(index)
            self.log_debug(f"Waypoint {index + 1} deleted")
            self.send_waypoints_to_server()
    
    def send_multiple_commands(self):
        """Send multiple commands - waypoints and start mission."""
        if self.current_view_mode != "pointcloud":
            self.log_debug("Cannot send waypoints: not in point cloud view")
            return
        
        try:
            waypoints = self.main_point_cloud.get_waypoints()
            added_waypoints = [wp for wp in waypoints if wp.get('added', False)]
            
            if len(added_waypoints) == 0:
                self.log_debug("No waypoints to send - Add waypoints first")
                return
            
            # Format waypoints sebagai string
            waypoints_strings = []
            for i, wp in enumerate(added_waypoints):
                pos_x, pos_y = wp['position']
                waypoint_str = f"[{pos_x:.2f},{pos_y:.2f},{wp['orientation']:.3f},{int(wp['yaw_enable'])},{int(wp['landing'])}]"
                waypoints_strings.append(waypoint_str)
            
            waypoints_data_str = ",".join(waypoints_strings)
            waypoints_command = f"coordinates [{waypoints_data_str}]"
            waypoints_success = self.send_websocket_command(waypoints_command)
            
            if waypoints_success:
                self.log_debug(f"Sent {len(waypoints_strings)} waypoints to server")
                
                time.sleep(0.1)  # Short delay
                start_success = self.send_websocket_command("start")
                if start_success:
                    self.log_debug("Mission started")
                else:
                    self.log_debug("Failed to start mission")
            else:
                self.log_debug("Failed to send waypoints - mission not started")
                
        except Exception as e:
            self.log_debug(f"Error in send_multiple_commands: {e}")
            print(f"Error sending multiple commands: {e}")
    
    def send_waypoints_to_server(self):
        """Send all added waypoints ke WebSocket server."""
        try:
            waypoints = self.main_point_cloud.get_waypoints()
            added_waypoints = [wp for wp in waypoints if wp.get('added', False)]
            
            if len(added_waypoints) == 0:
                return False
            
            waypoints_strings = []
            for wp in added_waypoints:
                pos_x, pos_y = wp['position']
                waypoint_str = f"[{pos_x:.2f},{pos_y:.2f},{wp['orientation']:.3f},{int(wp['yaw_enable'])},{int(wp['landing'])}]"
                waypoints_strings.append(waypoint_str)
            
            waypoints_data_str = ",".join(waypoints_strings)
            waypoints_command = f"coordinates [{waypoints_data_str}]"
            success = self.send_websocket_command(waypoints_command)
            
            if success:
                self.log_debug(f"Sent {len(waypoints_strings)} waypoints to server")
                return True
            else:
                self.log_debug(f"Failed to send waypoints to server")
                return False
                
        except Exception as e:
            self.log_debug(f"Error sending waypoints: {e}")
            return False
    
    def save_current_frame(self):
        """Save current point cloud frame."""
        if self.current_view_mode != "pointcloud":
            self.log_debug("Cannot save: not in point cloud view")
            return
            
        raw_points = self.main_point_cloud.raw_points
        if len(raw_points) == 0:
            self.log_debug("No frame to save")
            return
        
        import open3d as o3d
        timestamp = int(time.time())
        filename = f"pointcloud_raw_{timestamp}.ply"
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_points)
        
        raw_heights = raw_points[:, 2]
        raw_colors = self.main_point_cloud.generate_colors(raw_heights)
        if len(raw_colors) > 0:
            pcd.colors = o3d.utility.Vector3dVector(raw_colors)
        
        o3d.io.write_point_cloud(filename, pcd)
        self.log_debug(f"Saved: {filename} ({len(raw_points)} points)")
    
    # Utility methods
    def on_orientation_dial_changed(self, value):
        """Handle orientation dial change."""
        import math
        orientation = (value / 50.0 - 1.0) * math.pi
        self.ui.mcOrientation.setText(f"{orientation:.4f} rad")
    
    def update_altitude_display(self, value=None):
        """Update slider agar sesuai dengan nilai di QLabel DroneHeight (format 0.00 meter)."""
        try:
            # Ambil teks dari QLabel
            height_text = self.ui.DroneHeight.text().strip()

            # Jika label ada satuan 'm' atau 'meter', buang
            if height_text.endswith(" meter"):
                height_text = height_text.replace(" meter", "").strip()
            elif height_text.endswith(" m"):
                height_text = height_text.replace(" m", "").strip()

            # Konversi ke float (misalnya "12.34")
            altitude = float(height_text)

            # Ubah ke cm (slider integer)
            slider_value = int(round(altitude * 100))

            # Update slider
            self.ui.DroneAltitude.setValue(slider_value)

        except ValueError:
            # Kalau teks bukan angka valid, abaikan
            pass
    
    def log_debug(self, message):
        """Log message ke debugging console."""
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
    
    def closeEvent(self, event):
        """Clean shutdown with telemetry handler cleanup."""
        print("Shutting down Drone Control Center...")
        
        # Cleanup telemetry handler
        if hasattr(self, 'telemetry_handler') and self.telemetry_handler:
            self.telemetry_handler.cleanup()
        
        # Stop video stream
        if hasattr(self, 'video_stream') and hasattr(self.video_stream, 'stop_stream'):
            self.video_stream.stop_stream()
        
        # Stop TCP server
        if hasattr(self, 'tcp_receiver'):
            self.tcp_receiver.stop_server()
        if hasattr(self, 'tcp_thread') and self.tcp_thread:
            self.tcp_thread.quit()
            self.tcp_thread.wait()
        
        # Stop WebSocket client
        if hasattr(self, 'websocket_client'):
            self.websocket_client.stop_client()
        if hasattr(self, 'websocket_thread') and self.websocket_thread:
            self.websocket_thread.quit()
            self.websocket_thread.wait()
        
        # Stop drone parser
        if hasattr(self, 'drone_parser') and self.drone_parser:
            try:
                print("Stopping drone parser...")
                self.drone_parser.stop()
            except Exception as e:
                print(f"Error stopping drone parser: {e}")
        
        # Save waypoints
        if hasattr(self, 'main_point_cloud'):
            try:
                self.main_point_cloud.save_waypoints_to_json()
                print("Waypoints saved")
            except Exception as e:
                print(f"Error saving waypoints: {e}")
        
        event.accept()