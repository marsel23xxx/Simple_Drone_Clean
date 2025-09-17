"""
Main Window yang terintegrasi dengan desain UI yang sudah ada
Menggunakan activity_ui.py sebagai base dan menambahkan fungsionalitas lengkap
"""

import sys
import time
import threading
from pathlib import Path

from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QPixmap

# Import UI yang sudah dibuat
from activity_ui import Ui_MainWindow

# Import komponen fungsional
from config.settings import (
    APP_CONFIG, UI_CONFIG, ASSET_PATHS, NETWORK_CONFIG, FILE_PATHS
)
from ui.widgets.point_cloud_widget import SmoothPointCloudWidget
from ui.widgets.joystick_dialog import JoystickDialog
from core.tcp_receiver import TCPDataReceiver, TCPServerThread
from core.websocket_client import WebSocketCommandClient, WebSocketCommandThread
from core.drone_parser import DroneParser


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
        """Initialize communication components."""
        # TCP server for point cloud data
        self.tcp_receiver = TCPDataReceiver()
        self.tcp_thread = None
        
        # WebSocket client for commands
        self.websocket_client = WebSocketCommandClient()
        self.websocket_thread = None
        
        # Drone data parser
        try:
            self.drone_parser = DroneParser(port=NETWORK_CONFIG['drone_data_port'])
            self.drone_parser.start()
        except Exception as e:
            print(f"Failed to start drone parser: {e}")
            self.drone_parser = None
    
    def setup_functional_widgets(self):
        """Replace placeholder labels dengan functional widgets."""
        
        # Replace SwitchView_1 dengan main point cloud widget
        self.main_point_cloud = SmoothPointCloudWidget()
        self.main_point_cloud.setParent(self.ui.centralwidget)
        self.main_point_cloud.setGeometry(self.ui.SwitchView_1.geometry())
        self.main_point_cloud.setStyleSheet(self.ui.SwitchView_1.styleSheet())
        self.ui.SwitchView_1.hide()  # Hide original label
        
        # Replace SwitchView_2 dengan secondary point cloud widget  
        self.secondary_point_cloud = SmoothPointCloudWidget()
        self.secondary_point_cloud.setParent(self.ui.frame_8)
        self.secondary_point_cloud.setGeometry(self.ui.SwitchView_2.geometry())
        self.secondary_point_cloud.setStyleSheet(self.ui.SwitchView_2.styleSheet())
        self.ui.SwitchView_2.hide()  # Hide original label
        
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
        self.ui.mcSendCommand.clicked.connect(lambda: self.send_websocket_command("start"))
        self.ui.mcDialOrientation.valueChanged.connect(self.on_orientation_dial_changed)
        
        # Altitude slider
        self.ui.DroneAltitude.valueChanged.connect(self.update_altitude_display)
    
    def setup_timers(self):
        """Setup update timers."""
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(UI_CONFIG['update_intervals']['status_update'])
        
        # Drone data update timer
        if self.drone_parser:
            self.drone_timer = QTimer()
            self.drone_timer.timeout.connect(self.update_drone_data)
            self.drone_timer.start(UI_CONFIG['update_intervals']['drone_data'])
    
    def start_services(self):
        """Start communication services."""
        self.start_tcp_server()
        
    def start_tcp_server(self):
        """Start TCP server thread."""
        if self.tcp_thread is None or not self.tcp_thread.isRunning():
            self.tcp_thread = TCPServerThread(self.tcp_receiver)
            self.tcp_thread.start()
    
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
        else:
            print(f"Cannot send command '{command}': WebSocket not connected")
            self.log_debug(f"Cannot send '{command}': Not connected")
    
    def emergency_stop(self):
        """Emergency stop command."""
        self.send_websocket_command("stop")
        self.log_debug("EMERGENCY STOP ACTIVATED")
        print("EMERGENCY STOP ACTIVATED")
    
    def switch_views(self):
        """Switch between main dan secondary point cloud views."""
        # Swap content antara main dan secondary views
        # Get current point cloud data
        main_points = self.main_point_cloud.raw_points.copy()
        secondary_points = self.secondary_point_cloud.raw_points.copy()
        
        # Swap the data
        self.main_point_cloud.update_current_frame(secondary_points)
        self.secondary_point_cloud.update_current_frame(main_points)
        
        self.log_debug("Views switched")
    
    def toggle_edit_mode(self, state):
        """Toggle edit mode for waypoint management."""
        self.edit_mode = state == Qt.Checked
        self.main_point_cloud.set_edit_mode(self.edit_mode)
        self.secondary_point_cloud.set_edit_mode(self.edit_mode)
        
        # Update UI state
        if self.edit_mode:
            self.update_waypoints_table()
            self.log_debug("Edit mode enabled")
        else:
            self.log_debug("Edit mode disabled")
    
    def update_view_controls(self, state):
        """Update view controls."""
        top_down_enabled = state == Qt.Checked
        self.main_point_cloud.set_top_down_mode(top_down_enabled)
        self.secondary_point_cloud.set_top_down_mode(top_down_enabled)
        self.log_debug(f"Top-down view: {'Locked' if top_down_enabled else 'Unlocked'}")
    
    def update_height_filter(self, value):
        """Update height filtering."""
        self.main_point_cloud.set_z_filter(max_z=value, enabled=True)
        self.secondary_point_cloud.set_z_filter(max_z=value, enabled=True)
        self.log_debug(f"Height filter: {value:.1f}m")
    
    def display_frame(self, points):
        """Display new point cloud frame."""
        self.frame_count += 1
        
        # Calculate FPS
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        if time_diff > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / time_diff)
        self.last_frame_time = current_time
        
        # Update both point cloud widgets
        self.main_point_cloud.update_current_frame(points)
        self.secondary_point_cloud.update_current_frame(points)
    
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
    
    def update_drone_data(self):
        """Update drone data display."""
        if not self.drone_parser:
            return
            
        try:
            latest_data = self.drone_parser.get_latest()
            if not latest_data:
                return
                
            position = self.drone_parser.get_position(latest_data)
            rpy = self.drone_parser.get_rpy(latest_data)
            velocity = self.drone_parser.get_velocity(latest_data)
            battery = self.drone_parser.get_battery(latest_data)
            status = self.drone_parser.get_status(latest_data)
            
            if position:
                self.ui.DronePositionX.setText(f"[{position['x']:.2f}] m")
                self.ui.DronePositionY.setText(f"[{position['y']:.2f}] m")
                self.ui.DroneHeight.setText(f"{position.get('z', 0):.2f} meter")
                
                # Update point cloud widget dengan drone position
                if rpy:
                    self.main_point_cloud.update_drone_data(
                        position['x'], position['y'], rpy['yaw']
                    )
                    self.secondary_point_cloud.update_drone_data(
                        position['x'], position['y'], rpy['yaw']
                    )
            
            if velocity:
                self.ui.DroneSpeedX.setText(f"[{velocity['vx']:.2f}] m/s")
                self.ui.DroneSpeedY.setText(f"[{velocity['vy']:.2f}] m/s")
                self.ui.DroneSpeedZ.setText(f"[{velocity['vz']:.2f}] m/s")
            
            if battery:
                self.ui.DroneBattery.setValue(battery['percentage'])
                
                # Update battery color berdasarkan level
                if battery['percentage'] < 20:
                    self.ui.DroneBattery.setStyleSheet("""
                        QProgressBar::chunk { background-color: #e63946; }
                    """)
                elif battery['percentage'] < 50:
                    self.ui.DroneBattery.setStyleSheet("""
                        QProgressBar::chunk { background-color: #f77f00; }
                    """)
                else:
                    self.ui.DroneBattery.setStyleSheet("""
                        QProgressBar::chunk { background-color: #3ddb55; }
                    """)
            
            if status:
                self.ui.DroneMode.setText(status.get('mode', 'Auto'))
                if 'flight_time' in status:
                    minutes = status['flight_time'] // 60
                    seconds = status['flight_time'] % 60
                    self.ui.DroneFlightTime.setText(f"{minutes:02d}:{seconds:02d}")
            
        except Exception as e:
            pass  # Silently handle errors
    
    def update_status(self):
        """Update status display."""
        if self.frame_count > 0:
            status_text = f"LIVE - Display: {self.fps:.1f} FPS"
        else:
            status_text = "Waiting for point cloud data..."
        
        # Update command control status
        self.ui.CommandControl.setText("Active" if self.frame_count > 0 else "Idle")
    
    def update_waypoints_table(self):
        """Update waypoints display table."""
        waypoints = self.main_point_cloud.get_waypoints()
        self.ui.mcDisplayData.setRowCount(len(waypoints))
        
        for i, waypoint in enumerate(waypoints):
            pos_x, pos_y = waypoint['position']
            
            # Position
            from PyQt5.QtWidgets import QTableWidgetItem, QPushButton, QCheckBox
            self.ui.mcDisplayData.setItem(i, 0, 
                QTableWidgetItem(f"({pos_x:.2f}, {pos_y:.2f})"))
            
            # Orientation
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
        marker = self.main_point_cloud.get_marker()
        
        if marker:
            # Update position display
            x, y = marker['position']
            self.ui.scPositionX.setText(f"{x:.2f}")
            self.ui.scPositionY.setText(f"{y:.2f}")
            
            # Update orientation
            self.ui.scOrientation.setText(f"{marker['orientation']:.4f} rad")
            
            # Update checkboxes
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
    
    def edit_marker_orientation(self):
        """Edit marker orientation menggunakan joystick dialog."""
        if self.current_marker:
            current_orientation = self.current_marker['orientation']
            
            dialog = JoystickDialog(current_orientation, self)
            if dialog.exec_() == dialog.Accepted:
                new_orientation = dialog.orientation
                self.main_point_cloud.update_marker(orientation=new_orientation)
                self.log_debug(f"Marker orientation updated: {new_orientation:.3f} rad")
    
    def clear_marker(self):
        """Clear current marker."""
        self.main_point_cloud.clear_marker()
        self.log_debug("Marker cleared")
    
    def send_goto_command(self):
        """Send goto command dengan current marker data."""
        if self.current_marker:
            x, y = self.current_marker['position']
            orientation = self.current_marker['orientation']
            yaw_enable = 1 if self.current_marker['yaw_enable'] else 0
            landing = 1 if self.current_marker['landing'] else 0
            
            command = f"goto [{x:.2f}, {y:.2f}, {orientation:.3f}, {yaw_enable}, {landing}]"
            self.send_websocket_command(command)
            self.log_debug(f"Sent goto: {command}")
    
    def update_marker_yaw(self, state):
        """Update marker yaw enable state."""
        if self.current_marker:
            yaw_enable = state == Qt.Checked
            self.main_point_cloud.update_marker(yaw_enable=yaw_enable)
            self.log_debug(f"Marker yaw enable: {yaw_enable}")
    
    def update_marker_landing(self, state):
        """Update marker landing state."""
        if self.current_marker:
            landing = state == Qt.Checked
            self.main_point_cloud.update_marker(landing=landing)
            self.log_debug(f"Marker landing: {landing}")
    
    def edit_waypoint_orientation(self, index):
        """Edit waypoint orientation."""
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
        yaw_enable = state == Qt.Checked
        self.main_point_cloud.update_waypoint(index, yaw_enable=yaw_enable)
        self.log_debug(f"Waypoint {index + 1} yaw enable: {yaw_enable}")
    
    def update_waypoint_landing(self, index, state):
        """Update waypoint landing state."""
        landing = state == Qt.Checked
        self.main_point_cloud.update_waypoint(index, landing=landing)
        self.log_debug(f"Waypoint {index + 1} landing: {landing}")
    
    def add_waypoint(self, index):
        """Add waypoint."""
        self.main_point_cloud.update_waypoint(index, added=True)
        waypoints = self.main_point_cloud.get_waypoints()
        pos = waypoints[index]['position']
        self.log_debug(f"Waypoint {index + 1} added: ({pos[0]:.2f}, {pos[1]:.2f})")
    
    def delete_waypoint(self, index):
        """Delete waypoint."""
        self.main_point_cloud.delete_waypoint(index)
        self.log_debug(f"Waypoint {index + 1} deleted")
    
    def save_current_frame(self):
        """Save current point cloud frame."""
        raw_points = self.main_point_cloud.raw_points
        if len(raw_points) == 0:
            self.log_debug("No frame to save")
            return
        
        import open3d as o3d
        timestamp = int(time.time())
        filename = f"pointcloud_raw_{timestamp}.ply"
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_points)
        
        # Generate colors
        raw_heights = raw_points[:, 2]
        raw_colors = self.main_point_cloud.generate_colors(raw_heights)
        if len(raw_colors) > 0:
            pcd.colors = o3d.utility.Vector3dVector(raw_colors)
        
        o3d.io.write_point_cloud(filename, pcd)
        self.log_debug(f"Saved: {filename} ({len(raw_points)} points)")
    
    def on_orientation_dial_changed(self, value):
        """Handle orientation dial change."""
        # Convert dial value (0-99) ke radians (-π to π)
        import math
        orientation = (value / 50.0 - 1.0) * math.pi
        self.ui.mcOrientation.setText(f"{orientation:.4f} rad")
    
    def update_altitude_display(self, value):
        """Update altitude display."""
        # Update altitude meter display
        # This could be connected to drone altitude control
        pass
    
    def log_debug(self, message):
        """Log message ke debugging console."""
        import time
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
        """Clean shutdown."""
        print("Shutting down Drone Control Center...")
        
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
        
        # Stop drone parser
        if self.drone_parser:
            try:
                self.drone_parser.stop()
            except:
                pass
        
        event.accept()


if __name__ == "__main__":
    from ui.styles.dark_theme import apply_dark_theme
    
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    
    window = DroneControlMainWindow()
    window.show()
    
    sys.exit(app.exec_())