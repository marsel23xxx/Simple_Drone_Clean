"""
Command Panel Widget for Drone Control Center
Professional command interface with single and multiple command modes
"""

from PyQt5.QtWidgets import (
    QWidget, QFrame, QLabel, QPushButton,
    QCheckBox, QDoubleSpinBox, QDial, QTextBrowser
)
from PyQt5.QtCore import Qt, pyqtSignal, QRect
from PyQt5.QtGui import QFont

from config.settings import ASSET_PATHS


class CommandPanelWidget(QWidget):
    """Command panel with single and multiple command modes."""
    
    # Signals
    command_signal = pyqtSignal(str)  # Command to send
    edit_mode_changed = pyqtSignal(bool)  # Edit mode state
    height_filter_changed = pyqtSignal(float)  # Height filter value
    view_control_changed = pyqtSignal(bool)  # View control state
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_marker = None
        self.setup_ui()
        self.connect_signals()
    
    def setup_ui(self):
        """Setup the command panel UI."""
        self.setFixedSize(841, 361)
        self.setStyleSheet("""
            border-radius: 10px;
            border: 1px solid black;
            background-color: #4d4d4d;
        """)
        
        # Status indicator
        self.status_indicator = QLabel(self)
        self.status_indicator.setGeometry(QRect(10, 10, 31, 21))
        self.status_indicator.setStyleSheet("""
            background-color: rgb(0, 255, 0);
            border: 0px;
            border-radius: 10px;
        """)
        
        # Title
        title_label = QLabel("DRONE CONTROLLER", self)
        title_label.setGeometry(QRect(50, 10, 231, 21))
        title_label.setFont(QFont("Arial", 9, QFont.Bold))
        title_label.setStyleSheet("""
            color: rgb(156, 156, 156);
            font-weight: bold;
            border: 0px;
        """)
        
        # Main content area
        self.create_content_area()
    
    def create_content_area(self):
        """Create the main content area with command sections."""
        # Left section - Commands
        self.create_command_section()
        
        # Right section - Debugging
        self.create_debugging_section()
    
    def create_command_section(self):
        """Create command control section."""
        command_frame = QFrame(self)
        command_frame.setGeometry(QRect(10, 40, 571, 311))
        command_frame.setStyleSheet("""
            background-color: rgb(56, 56, 56);
            border-radius: 10px;
        """)
        
        # Single Command Section
        self.create_single_command_section(command_frame)
        
        # Multiple Command Section
        self.create_multiple_command_section(command_frame)
    
    def create_single_command_section(self, parent):
        """Create single command control section."""
        single_frame = QFrame(parent)
        single_frame.setGeometry(QRect(10, 20, 551, 81))
        
        # Section title
        title = QLabel("Single Command", parent)
        title.setGeometry(QRect(20, 10, 101, 21))
        title.setFont(QFont("Arial", 10, QFont.Bold))
        title.setStyleSheet("""
            color: rgb(148, 148, 148);
            border: 0px;
        """)
        title.setAlignment(Qt.AlignCenter)
        
        # Position display
        QLabel("Position :", single_frame).setGeometry(QRect(15, 20, 51, 21))
        QLabel("X :", single_frame).setGeometry(QRect(80, 20, 16, 21))
        QLabel("Y :", single_frame).setGeometry(QRect(160, 20, 16, 21))
        
        self.position_x_label = QLabel("0.00", single_frame)
        self.position_x_label.setGeometry(QRect(100, 20, 51, 21))
        self.position_y_label = QLabel("0.00", single_frame)
        self.position_y_label.setGeometry(QRect(180, 20, 51, 21))
        
        # Orientation display
        QLabel("Orientation :", single_frame).setGeometry(QRect(15, 50, 71, 21))
        self.orientation_label = QLabel("0.0000 rad", single_frame)
        self.orientation_label.setGeometry(QRect(100, 50, 71, 21))
        
        # Control buttons
        self.edit_button = QPushButton("Edit", single_frame)
        self.edit_button.setGeometry(QRect(230, 10, 101, 31))
        self.edit_button.clicked.connect(self.edit_marker_orientation)
        
        self.clear_button = QPushButton("Clear Marker", single_frame)
        self.clear_button.setGeometry(QRect(230, 45, 101, 31))
        self.clear_button.clicked.connect(self.clear_marker)
        
        self.hover_button = QPushButton("Hover", single_frame)
        self.hover_button.setGeometry(QRect(340, 10, 91, 31))
        self.hover_button.clicked.connect(lambda: self.command_signal.emit("hover"))
        
        self.send_goto_button = QPushButton("Send Goto", single_frame)
        self.send_goto_button.setGeometry(QRect(340, 45, 91, 31))
        self.send_goto_button.clicked.connect(self.send_goto_command)
        
        # Checkboxes
        self.yaw_enable_check = QCheckBox("Yaw Enable", single_frame)
        self.yaw_enable_check.setGeometry(QRect(440, 20, 101, 21))
        self.yaw_enable_check.stateChanged.connect(self.update_marker_yaw)
        
        self.landing_check = QCheckBox("Landing", single_frame)
        self.landing_check.setGeometry(QRect(440, 50, 101, 21))
        self.landing_check.stateChanged.connect(self.update_marker_landing)
        
        # Initially disable controls
        self.set_single_command_enabled(False)
    
    def create_multiple_command_section(self, parent):
        """Create multiple command control section."""
        multiple_frame = QFrame(parent)
        multiple_frame.setGeometry(QRect(10, 120, 551, 181))
        
        # Section title
        title = QLabel("Multiple Command", parent)
        title.setGeometry(QRect(20, 110, 111, 21))
        title.setFont(QFont("Arial", 10, QFont.Bold))
        title.setStyleSheet("""
            color: rgb(148, 148, 148);
            border: 0px;
        """)
        title.setAlignment(Qt.AlignCenter)
        
        # Edit Mode section
        QLabel("Edit Mode", multiple_frame).setGeometry(QRect(15, 16, 131, 21))
        self.edit_mode_check = QCheckBox("Edit Mode", multiple_frame)
        self.edit_mode_check.setGeometry(QRect(15, 40, 161, 21))
        self.edit_mode_check.stateChanged.connect(self.on_edit_mode_changed)
        
        # View Controls section
        QLabel("View Controls", multiple_frame).setGeometry(QRect(15, 66, 131, 21))
        self.top_down_check = QCheckBox("Lock to Top-Down View", multiple_frame)
        self.top_down_check.setGeometry(QRect(15, 90, 161, 21))
        self.top_down_check.setChecked(True)
        self.top_down_check.stateChanged.connect(self.on_view_control_changed)
        
        # Height Filtering section
        QLabel("Height Filtering (m) :", multiple_frame).setGeometry(QRect(15, 116, 131, 21))
        self.height_spinbox = QDoubleSpinBox(multiple_frame)
        self.height_spinbox.setGeometry(QRect(15, 140, 131, 31))
        self.height_spinbox.setRange(0.1, 10.0)
        self.height_spinbox.setSingleStep(0.1)
        self.height_spinbox.setValue(1.5)
        self.height_spinbox.setDecimals(1)
        self.height_spinbox.valueChanged.connect(self.on_height_filter_changed)
        
        # Command buttons
        self.home_button = QPushButton("Home", multiple_frame)
        self.home_button.setGeometry(QRect(190, 35, 186, 31))
        self.home_button.clicked.connect(lambda: self.command_signal.emit("home"))
        
        self.hover_multiple_button = QPushButton("Hover", multiple_frame)
        self.hover_multiple_button.setGeometry(QRect(190, 70, 186, 31))
        self.hover_multiple_button.clicked.connect(lambda: self.command_signal.emit("hover"))
        
        self.save_maps_button = QPushButton("Save Maps", multiple_frame)
        self.save_maps_button.setGeometry(QRect(190, 105, 186, 31))
        self.save_maps_button.clicked.connect(self.save_current_frame)
        
        self.start_autonomous_button = QPushButton("Start Autonomous", multiple_frame)
        self.start_autonomous_button.setGeometry(QRect(190, 140, 186, 31))
        self.start_autonomous_button.clicked.connect(lambda: self.command_signal.emit("start"))
        
        # Orientation control
        QLabel("Orientation :", multiple_frame).setGeometry(QRect(410, 10, 71, 21))
        self.orientation_dial_label = QLabel("0.0000 rad", multiple_frame)
        self.orientation_dial_label.setGeometry(QRect(485, 10, 61, 21))
        
        self.orientation_dial = QDial(multiple_frame)
        self.orientation_dial.setGeometry(QRect(395, 25, 151, 161))
        self.orientation_dial.setWrapping(True)
        self.orientation_dial.setNotchTarget(3.7)
        self.orientation_dial.valueChanged.connect(self.on_orientation_dial_changed)
        
        # Dial status indicator
        self.dial_status = QPushButton(multiple_frame)
        self.dial_status.setGeometry(QRect(510, 40, 10, 10))
        self.dial_status.setStyleSheet("""
            QPushButton {
                background-color: rgb(0, 255, 0);
                border: none;
                border-radius: 5px;
                min-width: 10px;
                min-height: 10px;
                max-width: 10px;
                max-height: 10px;
            }
        """)
    
    def create_debugging_section(self):
        """Create debugging console section."""
        debug_frame = QFrame(self)
        debug_frame.setGeometry(QRect(590, 40, 241, 311))
        debug_frame.setStyleSheet("""
            background-color: rgb(56, 56, 56);
            border-radius: 10px;
        """)
        
        # Debug title
        debug_title = QLabel("DEBUGGING CONSOLE", debug_frame)
        debug_title.setGeometry(QRect(10, 10, 221, 21))
        debug_title.setFont(QFont("Arial", 10, QFont.Bold))
        debug_title.setStyleSheet("""
            color: rgb(156, 156, 156);
            font-weight: bold;
            border: 0px;
        """)
        
        # Debug text browser
        self.debug_console = QTextBrowser(debug_frame)
        self.debug_console.setGeometry(QRect(10, 40, 221, 251))
        self.debug_console.setFont(QFont("Consolas", 8))
        self.debug_console.setStyleSheet("""
            color: white;
            border: 0px;
            background-color: #1e1e1e;
        """)
    
    def connect_signals(self):
        """Connect internal signals."""
        pass
    
    def set_single_command_enabled(self, enabled):
        """Enable/disable single command controls."""
        self.edit_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)
        self.send_goto_button.setEnabled(enabled)
        self.yaw_enable_check.setEnabled(enabled)
        self.landing_check.setEnabled(enabled)
    
    def update_marker_display(self, marker):
        """Update marker display information."""
        self.current_marker = marker
        
        if marker:
            # Update position display
            x, y = marker['position']
            self.position_x_label.setText(f"{x:.2f}")
            self.position_y_label.setText(f"{y:.2f}")
            
            # Update orientation display
            self.orientation_label.setText(f"{marker['orientation']:.4f} rad")
            
            # Update checkboxes
            self.yaw_enable_check.blockSignals(True)
            self.yaw_enable_check.setChecked(marker['yaw_enable'])
            self.yaw_enable_check.blockSignals(False)
            
            self.landing_check.blockSignals(True)
            self.landing_check.setChecked(marker['landing'])
            self.landing_check.blockSignals(False)
            
            # Enable controls
            self.set_single_command_enabled(True)
        else:
            # Clear display
            self.position_x_label.setText("0.00")
            self.position_y_label.setText("0.00")
            self.orientation_label.setText("0.0000 rad")
            
            # Clear checkboxes
            self.yaw_enable_check.blockSignals(True)
            self.yaw_enable_check.setChecked(False)
            self.yaw_enable_check.blockSignals(False)
            
            self.landing_check.blockSignals(True)
            self.landing_check.setChecked(False)
            self.landing_check.blockSignals(False)
            
            # Disable controls
            self.set_single_command_enabled(False)
    
    def edit_marker_orientation(self):
        """Edit marker orientation."""
        if self.current_marker:
            from ui.widgets.joystick_dialog import JoystickDialog
            
            current_orientation = self.current_marker['orientation']
            dialog = JoystickDialog(current_orientation, self)
            
            if dialog.exec_() == dialog.Accepted:
                new_orientation = dialog.orientation
                self.current_marker['orientation'] = new_orientation
                self.orientation_label.setText(f"{new_orientation:.4f} rad")
                
                # Signal parent to update marker
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(0, self.notify_marker_changed)
    
    def notify_marker_changed(self):
        """Notify parent that marker has changed."""
        if hasattr(self.parent(), 'main_point_cloud'):
            if self.current_marker:
                self.parent().main_point_cloud.update_marker(**self.current_marker)
    
    def clear_marker(self):
        """Clear current marker."""
        if hasattr(self.parent(), 'main_point_cloud'):
            self.parent().main_point_cloud.clear_marker()
    
    def send_goto_command(self):
        """Send goto command with current marker data."""
        if self.current_marker:
            x, y = self.current_marker['position']
            orientation = self.current_marker['orientation']
            yaw_enable = 1 if self.current_marker['yaw_enable'] else 0
            landing = 1 if self.current_marker['landing'] else 0
            
            command = f"goto [{x:.2f}, {y:.2f}, {orientation:.3f}, {yaw_enable}, {landing}]"
            self.command_signal.emit(command)
            self.log_debug(f"Sent: {command}")
    
    def update_marker_yaw(self, state):
        """Update marker yaw enable state."""
        if self.current_marker:
            self.current_marker['yaw_enable'] = state == Qt.Checked
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, self.notify_marker_changed)
    
    def update_marker_landing(self, state):
        """Update marker landing state."""
        if self.current_marker:
            self.current_marker['landing'] = state == Qt.Checked
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, self.notify_marker_changed)
    
    def on_edit_mode_changed(self, state):
        """Handle edit mode change."""
        enabled = state == Qt.Checked
        self.edit_mode_changed.emit(enabled)
        self.log_debug(f"Edit mode: {'Enabled' if enabled else 'Disabled'}")
    
    def on_view_control_changed(self, state):
        """Handle view control change."""
        enabled = state == Qt.Checked
        self.view_control_changed.emit(enabled)
        self.log_debug(f"Top-down view: {'Locked' if enabled else 'Unlocked'}")
    
    def on_height_filter_changed(self, value):
        """Handle height filter change."""
        self.height_filter_changed.emit(value)
        self.log_debug(f"Height filter: {value:.1f}m")
    
    def on_orientation_dial_changed(self, value):
        """Handle orientation dial change."""
        # Convert dial value (0-99) to radians (-π to π)
        orientation = (value / 50.0 - 1.0) * 3.14159
        self.orientation_dial_label.setText(f"{orientation:.4f} rad")
    
    def save_current_frame(self):
        """Save current point cloud frame."""
        if hasattr(self.parent(), 'main_point_cloud'):
            # Get raw points from main point cloud widget
            raw_points = self.parent().main_point_cloud.raw_points
            if len(raw_points) > 0:
                import time
                import open3d as o3d
                
                timestamp = int(time.time())
                filename = f"pointcloud_raw_{timestamp}.ply"
                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(raw_points)
                
                # Generate colors
                raw_heights = raw_points[:, 2]
                raw_colors = self.parent().main_point_cloud.generate_colors(raw_heights)
                if len(raw_colors) > 0:
                    pcd.colors = o3d.utility.Vector3dVector(raw_colors)
                
                o3d.io.write_point_cloud(filename, pcd)
                self.log_debug(f"Saved: {filename} ({len(raw_points)} points)")
            else:
                self.log_debug("No point cloud data to save")
    
    def log_debug(self, message):
        """Log message to debug console."""
        import time
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        self.debug_console.append(formatted_message)
        
        # Keep only last 100 lines
        if self.debug_console.document().blockCount() > 100:
            cursor = self.debug_console.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()  # Remove the newline
    
    def update_status(self, status_text):
        """Update status display."""
        # You can add status display logic here if needed
        pass