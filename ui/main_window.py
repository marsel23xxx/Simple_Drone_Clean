"""
ui/main_window.py - CLEAN VERSION with Enhanced Multi-Camera AI Integration

Drone Control Center Main Window
Features:
- Independent multi-camera operation (if one fails, others continue)
- Full AI detection suite (Landolt 3D, QR 3D, Hazmat, Crack, Rust)
- Graceful degradation and error handling
"""

import sys
import time
import os
from pathlib import Path
from PyQt5.QtWidgets import (QMainWindow, QApplication, QLabel, QTableWidgetItem, 
                             QPushButton, QCheckBox)
from PyQt5.QtCore import QTimer, Qt

# Import UI
from activity_ui import Ui_MainWindow

# Import settings
from config.settings import (APP_CONFIG, UI_CONFIG, ASSET_PATHS, 
                            NETWORK_CONFIG, FILE_PATHS)

# Import widgets
from .widgets.point_cloud_widget import SmoothPointCloudWidget
from .widgets.video_stream_widget import RTSPStreamWidget, VideoStreamWidget
from .widgets.joystick_dialog import JoystickDialog

# Import core components
from core.tcp_receiver import TCPDataReceiver, TCPServerThread
from core.websocket_client import WebSocketCommandClient, WebSocketCommandThread
from core.drone_parser import DroneParser
from .widgets.drone_telemetry_handler import DroneTelemetryHandler

# Import AI system
from ai.detectors.my_detection import MultiCameraController


class DroneControlMainWindow(QMainWindow):
    """Main Window with Enhanced Multi-Camera AI Integration"""
    
    def __init__(self):
        super().__init__()
        
        # Setup UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Initialize state variables
        self._init_state_variables()
        
        # Setup components in order
        self.setup_multi_camera_ai()
        self.setup_communication()
        self.setup_functional_widgets()
        self.connect_signals()
        self.setup_timers()
        self.update_asset_paths()
        
        # Start services
        self.start_services()
        
        # Initial view
        self.switch_views()
        
        print("\n✅ Drone Control Center initialized successfully\n")
    
    # ========================================
    # INITIALIZATION
    # ========================================
    
    def _init_state_variables(self):
        """Initialize all state variables"""
        # AI state
        self.ai_controller = None
        self.ai_enabled = True
        self.current_ai_mode = 'standby'
        
        # Frame tracking
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0.0
        
        # View state
        self.current_view_mode = "pointcloud"
        self.current_marker = None
        self.edit_mode = False
        
        # Camera frame storage
        self.cam0_latest_frame = None
        self.cam1_latest_frame = None
        self.cam2_latest_frame = None
        
        # FPS monitoring
        self.frame_counters = {'main': 0, 'cam1': 0, 'cam2': 0}
        self.last_fps_time = time.time()
    
    def setup_multi_camera_ai(self):
        """Initialize Enhanced Multi-Camera AI Controller"""
        try:
            print("\n" + "="*60)
            print("🤖 INITIALIZING MULTI-CAMERA AI SYSTEM")
            print("="*60)
            
            # Model paths
            crack_weights = 'ai/models/crack.pt'
            hazmat_weights = 'ai/models/hazmat.pt'
            rust_model = 'ai/models/deeplabv3_corrosion_multiclass.pth'
            
            # Check if at least one model exists
            if not any([os.path.exists(p) for p in [crack_weights, hazmat_weights, rust_model]]):
                print("❌ No model files found - AI disabled")
                self.ai_enabled = False
                return
            
            # Initialize controller
            self.ai_controller = MultiCameraController(
                num_cameras=3,
                crack_weights=crack_weights,
                hazmat_weights=hazmat_weights,
                rust_model=rust_model
            )
            
            print("✅ Multi-Camera AI Controller initialized")
            print(f"   Mode: {self.ai_controller.current_mode.upper()}")
            print(f"   Cameras: {self.ai_controller.num_cameras}")
            print(f"   Features: Landolt 3D, QR 3D, Hazmat, Crack, Rust (Full)")
            print("="*60 + "\n")
            
            self._print_ai_controls()
            self.ai_enabled = True
            
        except Exception as e:
            print(f"❌ AI initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.ai_enabled = False
    
    def _print_ai_controls(self):
        """Print AI keyboard controls"""
        print("💡 AI Mode Controls:")
        print("   Q - QR Detection (3D + Auto-Save)")
        print("   W - Hazmat Detection")
        print("   E - Crack Detection")
        print("   R - Rust Detection (Full with Position)")
        print("   T - Landolt Detection (3D Analytics)")
        print("   Y - Motion Detection")
        print("   J/K/L - Save Camera 0/1/2")
        print("   P - Discard all & STANDBY\n")
    
    def setup_communication(self):
        """Initialize communication components"""
        # TCP server for point cloud
        self.tcp_receiver = TCPDataReceiver()
        self.tcp_thread = None
        
        # WebSocket client for commands
        self.websocket_client = WebSocketCommandClient()
        self.websocket_thread = None
        
        # UDP telemetry
        self.drone_parser = None
        self.telemetry_handler = None
        
        self._init_drone_parser()
    
    def _init_drone_parser(self):
        """Initialize drone parser for UDP telemetry"""
        try:
            drone_port = 8889
            
            try:
                self.drone_parser = DroneParser(port=drone_port, max_records=1000)
            except TypeError:
                self.drone_parser = DroneParser(port=drone_port)
            
            self.telemetry_handler = DroneTelemetryHandler(self, self.drone_parser)
            self.drone_parser.start(callback=self.telemetry_handler.on_udp_packet_received)
            
            print(f"✅ DroneParser started on port {drone_port}")
            
        except Exception as e:
            print(f"⚠️ DroneParser initialization failed: {e}")
            self.drone_parser = None
    
    def setup_functional_widgets(self):
        """Setup functional widgets"""
        # Point cloud widget
        self.main_point_cloud = SmoothPointCloudWidget()
        self.main_point_cloud.setParent(self.ui.centralwidget)
        self.main_point_cloud.setGeometry(self.ui.SwitchView_1.geometry())
        self.main_point_cloud.setStyleSheet(self.ui.SwitchView_1.styleSheet())
        self.ui.SwitchView_1.hide()
        
        # Main video stream (Camera 0)
        self.video_stream = RTSPStreamWidget()
        self.video_stream.setParent(self.ui.frame_8)
        self.video_stream.setGeometry(self.ui.SwitchView_2.geometry())
        self.video_stream.setStyleSheet(self.ui.SwitchView_2.styleSheet())
        self.ui.SwitchView_2.hide()
        
        # Camera 1 widget (Bottom)
        if hasattr(self.ui, 'imgDetector'):
            self.camera1_widget = VideoStreamWidget()
            self.camera1_widget.setParent(self.ui.imgDetector.parent())
            self.camera1_widget.setGeometry(self.ui.imgDetector.geometry())
            self.camera1_widget.setStyleSheet(self.ui.imgDetector.styleSheet())
            self.ui.imgDetector.hide()
        
        # Camera 2 widget (Top)
        if hasattr(self.ui, 'imgCapture'):
            self.camera2_widget = VideoStreamWidget()
            self.camera2_widget.setParent(self.ui.imgCapture.parent())
            self.camera2_widget.setGeometry(self.ui.imgCapture.geometry())
            self.camera2_widget.setStyleSheet(self.ui.imgCapture.styleSheet())
            self.ui.imgCapture.hide()
        
        # Initial visibility
        self.main_point_cloud.show()
        self.video_stream.hide()
    
    def connect_signals(self):
        """Connect all signals"""
        # TCP signals
        self.tcp_receiver.frame_received.connect(self.display_frame)
        self.tcp_receiver.connection_status.connect(self.update_tcp_status)
        
        # WebSocket signals
        self.websocket_client.connection_status.connect(self.update_websocket_status)
        
        # Point cloud signals
        self.main_point_cloud.waypoint_added.connect(self.update_waypoints_table)
        self.main_point_cloud.waypoint_changed.connect(self.update_waypoints_table)
        self.main_point_cloud.marker_changed.connect(self.update_marker_display)
        
        # Video stream signals
        self.video_stream.frame_received.connect(self.update_video_status)
        
        # UI buttons
        self.ui.CommandConnect.clicked.connect(self.toggle_connection)
        self.ui.btAutonomousEmergency.clicked.connect(self.emergency_stop)
        self.ui.DroneSwitch.clicked.connect(self.switch_views)
        
        if hasattr(self.ui, 'btVideoStream'):
            self.ui.btVideoStream.clicked.connect(self.toggle_video_stream)
        if hasattr(self.ui, 'DroneRefreshVideo'):
            self.ui.DroneRefreshVideo.clicked.connect(self.refresh_video_streams)
        
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
        """Setup update timers"""
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.start(UI_CONFIG['update_intervals']['status_update'])
        
        # AI processing timer (30 FPS)
        if self.ai_enabled and self.ai_controller:
            self.ai_timer = QTimer()
            self.ai_timer.timeout.connect(self.process_ai_frames)
            self.ai_timer.start(33)
        
        # FPS monitor timer
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps_display)
        self.fps_timer.start(2000)
    
    def update_asset_paths(self):
        """Update asset paths for styling"""
        import re
        
        widgets_with_assets = [
            (self.ui.label_62, 'compass'),
            (self.ui.label_60, 'compass'),
            (self.ui.label_61, 'compass'),
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
                asset_path = str(ASSET_PATHS[asset_key]).replace('\\', '/')
                if 'image: url(' in current_style:
                    new_style = re.sub(
                        r'image: url\(["\']?[^"\']*["\']?\);',
                        f'image: url("{asset_path}");',
                        current_style
                    )
                    widget.setStyleSheet(new_style)
    
    # ========================================
    # SERVICE MANAGEMENT
    # ========================================
    
    def start_services(self):
        """Start all communication services"""
        self.start_tcp_server()
        self.start_video_stream()
    
    def start_tcp_server(self):
        """Start TCP server thread"""
        if self.tcp_thread is None or not self.tcp_thread.isRunning():
            self.tcp_thread = TCPServerThread(self.tcp_receiver)
            self.tcp_thread.start()
    
    def start_video_stream(self, stream_url=None, cam1_url=None, cam2_url=None,
                          width_scale=0.5, height_scale=0.5):
        """Start independent multi-camera streams"""
        # Default URLs
        if not stream_url:
            stream_url = "rtsp://192.168.1.99:1234"
        if not cam1_url:
            cam1_url = "rtsp://192.168.1.88:8555/bottom"
        if not cam2_url:
            cam2_url = "rtsp://192.168.1.88:8554/top"
        
        try:
            print("\n" + "="*60)
            print("🎬 STARTING MULTI-CAMERA STREAMS")
            print("="*60)
            
            # Set AI controller
            if self.ai_enabled and self.ai_controller:
                self.video_stream.set_ai_controller(self.ai_controller)
                print("✅ AI controller set")
            
            successful_cameras = []
            
            # Start Camera 0 (Main)
            if self._start_camera("cam0", stream_url, width_scale, height_scale, 
                                self.on_cam0_frame):
                successful_cameras.append("Camera 0 (Main)")
            
            time.sleep(0.3)
            
            # Start Camera 1 (Bottom)
            if self._start_camera("cam1", cam1_url, width_scale, height_scale,
                                self.on_cam1_frame):
                successful_cameras.append("Camera 1 (Bottom)")
            
            time.sleep(0.3)
            
            # Start Camera 2 (Top)
            if self._start_camera("cam2", cam2_url, width_scale, height_scale,
                                self.on_cam2_frame):
                successful_cameras.append("Camera 2 (Top)")
            
            print("="*60)
            if successful_cameras:
                msg = f"✅ {len(successful_cameras)}/3 cameras active: {', '.join(successful_cameras)}"
                print(msg)
                self.log_debug(msg)
            else:
                print("❌ No cameras started")
                self.log_debug("Failed to start any cameras")
            print("="*60 + "\n")
            
            return len(successful_cameras) > 0
            
        except Exception as e:
            print(f"❌ Critical error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _start_camera(self, cam_id, url, width_scale, height_scale, frame_callback):
        """Helper to start individual camera"""
        try:
            print(f"📹 Starting {cam_id}: {url[:50]}...")
            
            success = self.video_stream.start_single_camera(
                cam_id, url, width_scale, height_scale, self.ai_enabled
            )
            
            if success and cam_id in self.video_stream.cameras:
                self.video_stream.cameras[cam_id].frame_ready.connect(frame_callback)
                print(f"   ✅ {cam_id} started")
                return True
            else:
                print(f"   ❌ {cam_id} failed")
                return False
                
        except Exception as e:
            print(f"   ❌ {cam_id} error: {e}")
            return False
    
    def stop_video_stream(self):
        """Stop all video streams"""
        try:
            self.video_stream.stop_stream()
            self.log_debug("Video streams stopped")
        except Exception as e:
            self.log_debug(f"Error stopping streams: {e}")
    
    def toggle_video_stream(self):
        """Toggle video streams on/off"""
        has_active = (hasattr(self.video_stream, 'cameras') and 
                     len(self.video_stream.cameras) > 0)
        
        if has_active:
            self.stop_video_stream()
        else:
            self.start_video_stream()
    
    def refresh_video_streams(self):
        """Refresh all video streams"""
        try:
            self.log_debug("Refreshing streams...")
            print("\n🔄 REFRESHING STREAMS")
            
            self.stop_video_stream()
            time.sleep(1)
            
            success = self.start_video_stream()
            
            if success:
                self.log_debug("✅ Streams refreshed")
                print("✅ Streams refreshed\n")
            else:
                self.log_debug("❌ Refresh failed")
                print("❌ Refresh failed\n")
                
        except Exception as e:
            error_msg = f"Refresh error: {e}"
            self.log_debug(error_msg)
            print(f"❌ {error_msg}\n")
    
    # ========================================
    # AI PROCESSING
    # ========================================
    
    def on_cam0_frame(self, frame):
        """Store Camera 0 frame"""
        self.cam0_latest_frame = frame.copy()
    
    def on_cam1_frame(self, frame):
        """Store Camera 1 frame"""
        self.cam1_latest_frame = frame.copy()
    
    def on_cam2_frame(self, frame):
        """Store Camera 2 frame"""
        self.cam2_latest_frame = frame.copy()
    
    def process_ai_frames(self):
        """Process frames with AI - gracefully handles missing cameras"""
        if not self.ai_enabled or not self.ai_controller:
            return
        
        # Get available frames
        frames = [self.cam0_latest_frame, self.cam1_latest_frame, self.cam2_latest_frame]
        available_count = sum(1 for f in frames if f is not None)
        
        if available_count == 0:
            return
        
        try:
            import numpy as np
            
            # Create dummy frame for missing cameras
            template = next((f for f in frames if f is not None), None)
            if template is None:
                return
            
            dummy_frame = np.zeros_like(template)
            
            # Fill missing frames
            frame0 = frames[0] if frames[0] is not None else dummy_frame
            frame1 = frames[1] if frames[1] is not None else dummy_frame
            frame2 = frames[2] if frames[2] is not None else dummy_frame
            
            # Process all frames
            processed_frames = self.ai_controller.process_frames(frame0, frame1, frame2)
            
            if len(processed_frames) == 3:
                # Update displays for active cameras only
                if self.cam0_latest_frame is not None:
                    try:
                        self.video_stream.update_frame(processed_frames[0])
                    except:
                        pass
                
                if self.cam1_latest_frame is not None and hasattr(self, 'camera1_widget'):
                    try:
                        self.camera1_widget.update_frame(processed_frames[1])
                    except:
                        pass
                
                if self.cam2_latest_frame is not None and hasattr(self, 'camera2_widget'):
                    try:
                        self.camera2_widget.update_frame(processed_frames[2])
                    except:
                        pass
        
        except Exception as e:
            # Log occasionally to avoid spam
            if not hasattr(self, '_last_ai_error_time'):
                self._last_ai_error_time = 0
            
            current_time = time.time()
            if current_time - self._last_ai_error_time > 5.0:
                print(f"AI processing error: {e}")
                self._last_ai_error_time = current_time
    
    def switch_ai_mode(self, mode):
        """Switch AI detection mode"""
        if not self.ai_controller:
            return
        
        mode_map = {
            'qr': self.ai_controller.MODES['QR'],
            'hazmat': self.ai_controller.MODES['HAZMAT'],
            'crack': self.ai_controller.MODES['CRACK'],
            'rust': self.ai_controller.MODES['RUST'],
            'landolt': self.ai_controller.MODES['LANDOLT'],
            'motion': self.ai_controller.MODES['MOTION'],
            'standby': self.ai_controller.MODES['STANDBY']
        }
        
        if mode in mode_map:
            self.ai_controller.switch_mode(mode_map[mode])
            self.current_ai_mode = mode
            self.log_debug(f"AI Mode: {mode.upper()}")
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for AI control"""
        if not self.ai_controller:
            super().keyPressEvent(event)
            return
        
        key = event.key()
        
        # AI mode switches
        mode_keys = {
            Qt.Key_Q: 'qr',
            Qt.Key_W: 'hazmat',
            Qt.Key_E: 'crack',
            Qt.Key_R: 'rust',
            Qt.Key_T: 'landolt',
            Qt.Key_Y: 'motion'
        }
        
        if key in mode_keys:
            self.switch_ai_mode(mode_keys[key])
            return
        
        # Save commands
        if key == Qt.Key_J:
            self._save_camera(0)
        elif key == Qt.Key_K:
            self._save_camera(1)
        elif key == Qt.Key_L:
            self._save_camera(2)
        elif key == Qt.Key_P:
            # Discard all and return to standby
            for worker in self.ai_controller.workers:
                worker.unfreeze()
            self.switch_ai_mode('standby')
            self.log_debug("All cameras unfrozen - STANDBY")
        else:
            super().keyPressEvent(event)
    
    def _save_camera(self, cam_index):
        """Helper to save camera detection"""
        if cam_index >= len(self.ai_controller.workers):
            return
        
        worker = self.ai_controller.workers[cam_index]
        if worker.is_frozen:
            worker.save_current_detection(self.current_ai_mode)
            self.log_debug(f"Camera {cam_index} saved")
        else:
            self.log_debug(f"Camera {cam_index} not frozen - cannot save")
    
    # ========================================
    # WEBSOCKET COMMANDS
    # ========================================
    
    def toggle_connection(self):
        """Toggle WebSocket connection"""
        if self.websocket_client.connected:
            self.disconnect_websocket()
        else:
            self.connect_websocket()
    
    def connect_websocket(self):
        """Connect to WebSocket server"""
        if self.websocket_thread is None or not self.websocket_thread.isRunning():
            self.websocket_client.start_client()
            self.websocket_thread = WebSocketCommandThread(self.websocket_client)
            self.websocket_thread.start()
    
    def disconnect_websocket(self):
        """Disconnect from WebSocket server"""
        self.websocket_client.stop_client()
        if self.websocket_thread:
            self.websocket_thread.quit()
            self.websocket_thread.wait()
            self.websocket_thread = None
    
    def send_websocket_command(self, command):
        """Send command through WebSocket"""
        if self.websocket_thread and self.websocket_client.connected:
            success = self.websocket_thread.send_command_sync(command)
            if not success:
                self.log_debug(f"Failed to send: {command}")
            return success
        else:
            self.log_debug(f"Cannot send '{command}': Not connected")
            return False
    
    def emergency_stop(self):
        """Emergency stop command"""
        self.send_websocket_command("stop")
        self.log_debug("🚨 EMERGENCY STOP ACTIVATED")
        print("🚨 EMERGENCY STOP ACTIVATED")
    
    # ========================================
    # VIEW MANAGEMENT
    # ========================================
    
    def switch_views(self):
        """Swap point cloud and video positions"""
        # Get current properties
        pc_geometry = self.main_point_cloud.geometry()
        pc_parent = self.main_point_cloud.parent()
        pc_style = self.main_point_cloud.styleSheet()
        
        video_geometry = self.video_stream.geometry()
        video_parent = self.video_stream.parent()
        video_style = self.video_stream.styleSheet()
        
        # Swap
        self.main_point_cloud.setParent(video_parent)
        self.main_point_cloud.setGeometry(video_geometry)
        self.main_point_cloud.setStyleSheet(video_style)
        
        self.video_stream.setParent(pc_parent)
        self.video_stream.setGeometry(pc_geometry)
        self.video_stream.setStyleSheet(pc_style)
        
        # Show both
        self.main_point_cloud.show()
        self.video_stream.show()
        
        # Toggle mode
        if self.current_view_mode == "pointcloud":
            self.current_view_mode = "video"
            self.ui.DroneSwitch.setText("Point Cloud Main")
            self.ui.mcEditMode.setEnabled(False)
            self.ui.scEdit.setEnabled(False)
            self.ui.scSendGoto.setEnabled(False)
            self.ui.scClearMarker.setEnabled(False)
            self.log_debug("View: Video main, Point cloud secondary")
        else:
            self.current_view_mode = "pointcloud"
            self.ui.DroneSwitch.setText("Video Main")
            self.ui.mcEditMode.setEnabled(True)
            self.update_marker_display()
            self.log_debug("View: Point cloud main, Video secondary")
    
    def display_frame(self, points):
        """Display new point cloud frame"""
        self.frame_count += 1
        
        # Calculate FPS
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        if time_diff > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / time_diff)
        self.last_frame_time = current_time
        
        # Update widget
        self.main_point_cloud.update_current_frame(points)
    
    def update_video_status(self):
        """Update video stream status"""
        pass
    
    # ========================================
    # STATUS UPDATES
    # ========================================
    
    def update_tcp_status(self, connected, message):
        """Update TCP connection status"""
        color = "rgb(0, 255, 0)" if connected else "rgb(255, 0, 0)"
        text = "Connected" if connected else "Disconnected"
        
        self.ui.CommandOnline.setStyleSheet(f"""
            background-color: {color};
            border: 0px;
            border-radius: 10px;
        """)
        self.ui.CommandStatus.setText(text)
        self.ui.btQrcodeOnline_3.setStyleSheet(f"""
            background-color: {color};
            border: 0px;
        """)
    
    def update_websocket_status(self, connected, message):
        """Update WebSocket connection status"""
        if connected:
            self.ui.CommandConnect.setText("DISCONNECT")
            self.ui.CommandMode.setText("Connected")
        else:
            self.ui.CommandConnect.setText("CONNECT")
            self.ui.CommandMode.setText("Disconnected")
    
    # ========================================
    # WAYPOINT MANAGEMENT (abbreviated for clarity)
    # ========================================
    
    def toggle_edit_mode(self, state):
        """Toggle waypoint edit mode"""
        if self.current_view_mode != "pointcloud":
            self.ui.mcEditMode.setChecked(False)
            return
        
        self.edit_mode = state == Qt.Checked
        self.main_point_cloud.set_edit_mode(self.edit_mode)
        self.log_debug(f"Edit mode: {'enabled' if self.edit_mode else 'disabled'}")
        
        if self.edit_mode:
            self.update_waypoints_table()
    
    def update_view_controls(self, state):
        """Update view controls"""
        if self.current_view_mode == "pointcloud":
            top_down = state == Qt.Checked
            self.main_point_cloud.set_top_down_mode(top_down)
            self.log_debug(f"Top-down: {'locked' if top_down else 'unlocked'}")
    
    def update_height_filter(self, value):
        """Update height filtering"""
        if self.current_view_mode == "pointcloud":
            self.main_point_cloud.set_z_filter(max_z=value, enabled=True)
            self.log_debug(f"Height filter: {value:.1f}m")
    
    def update_waypoints_table(self):
        """Update waypoints table"""
        if self.current_view_mode != "pointcloud":
            return
        
        waypoints = self.main_point_cloud.get_waypoints()
        self.ui.mcDisplayData.setRowCount(len(waypoints))
        
        for i, wp in enumerate(waypoints):
            pos_x, pos_y = wp['position']
            
            # Position
            self.ui.mcDisplayData.setItem(i, 0, 
                QTableWidgetItem(f"({pos_x:.2f}, {pos_y:.2f})"))
            
            # Orientation
            self.ui.mcDisplayData.setItem(i, 1,
                QTableWidgetItem(f"{wp['orientation']:.3f}"))
            
            # Edit button
            edit_btn = QPushButton("Edit")
            edit_btn.clicked.connect(lambda _, idx=i: self.edit_waypoint_orientation(idx))
            self.ui.mcDisplayData.setCellWidget(i, 2, edit_btn)
            
            # Yaw Enable
            yaw_check = QCheckBox()
            yaw_check.setChecked(wp['yaw_enable'])
            yaw_check.stateChanged.connect(lambda s, idx=i: self.update_waypoint_yaw(idx, s))
            self.ui.mcDisplayData.setCellWidget(i, 3, yaw_check)
            
            # Landing
            land_check = QCheckBox()
            land_check.setChecked(wp['landing'])
            land_check.stateChanged.connect(lambda s, idx=i: self.update_waypoint_landing(idx, s))
            self.ui.mcDisplayData.setCellWidget(i, 4, land_check)
            
            # Action
            if wp['added']:
                action_btn = QPushButton("Delete")
                action_btn.clicked.connect(lambda _, idx=i: self.delete_waypoint(idx))
            else:
                action_btn = QPushButton("Add")
                action_btn.clicked.connect(lambda _, idx=i: self.add_waypoint(idx))
            self.ui.mcDisplayData.setCellWidget(i, 5, action_btn)
    
    def update_marker_display(self):
        """Update marker display"""
        if self.current_view_mode != "pointcloud":
            self._clear_marker_display()
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
            
            self.ui.scEdit.setEnabled(True)
            self.ui.scSendGoto.setEnabled(True)
            self.ui.scClearMarker.setEnabled(True)
            
            self.current_marker = marker
        else:
            self._clear_marker_display()
    
    def _clear_marker_display(self):
        """Clear marker display"""
        self.ui.scPositionX.setText("0.00")
        self.ui.scPositionY.setText("0.00")
        self.ui.scOrientation.setText("0.0000 rad")
        self.ui.scYawEnable.setChecked(False)
        self.ui.scLanding.setChecked(False)
        self.ui.scEdit.setEnabled(False)
        self.ui.scSendGoto.setEnabled(False)
        self.ui.scClearMarker.setEnabled(False)
        self.current_marker = None
    
    # ========================================
    # MARKER ACTIONS
    # ========================================
    
    def edit_marker_orientation(self):
        """Edit marker orientation"""
        if self.current_marker and self.current_view_mode == "pointcloud":
            dialog = JoystickDialog(self.current_marker['orientation'], self)
            if dialog.exec_() == dialog.Accepted:
                self.main_point_cloud.update_marker(orientation=dialog.orientation)
                self.log_debug(f"Marker orientation: {dialog.orientation:.3f} rad")
    
    def clear_marker(self):
        """Clear current marker"""
        if self.current_view_mode == "pointcloud":
            self.main_point_cloud.clear_marker()
            self.log_debug("Marker cleared")
    
    def send_goto_command(self):
        """Send goto command"""
        if self.current_marker and self.current_view_mode == "pointcloud":
            x, y = self.current_marker['position']
            ori = self.current_marker['orientation']
            yaw = 1 if self.current_marker['yaw_enable'] else 0
            land = 1 if self.current_marker['landing'] else 0
            
            cmd = f"goto [{x:.2f}, {y:.2f}, {ori:.3f}, {yaw}, {land}]"
            self.send_websocket_command(cmd)
            self.log_debug(f"Sent: {cmd}")
    
    def update_marker_yaw(self, state):
        """Update marker yaw enable"""
        if self.current_marker and self.current_view_mode == "pointcloud":
            yaw_enable = state == Qt.Checked
            self.main_point_cloud.update_marker(yaw_enable=yaw_enable)
    
    def update_marker_landing(self, state):
        """Update marker landing"""
        if self.current_marker and self.current_view_mode == "pointcloud":
            landing = state == Qt.Checked
            self.main_point_cloud.update_marker(landing=landing)
    
    # ========================================
    # WAYPOINT ACTIONS
    # ========================================
    
    def edit_waypoint_orientation(self, index):
        """Edit waypoint orientation"""
        if self.current_view_mode != "pointcloud":
            return
        
        waypoints = self.main_point_cloud.get_waypoints()
        if 0 <= index < len(waypoints):
            dialog = JoystickDialog(waypoints[index]['orientation'], self)
            if dialog.exec_() == dialog.Accepted:
                self.main_point_cloud.update_waypoint(index, orientation=dialog.orientation)
                self.log_debug(f"Waypoint {index+1} orientation: {dialog.orientation:.3f} rad")
    
    def update_waypoint_yaw(self, index, state):
        """Update waypoint yaw enable"""
        if self.current_view_mode == "pointcloud":
            self.main_point_cloud.update_waypoint(index, yaw_enable=(state == Qt.Checked))
    
    def update_waypoint_landing(self, index, state):
        """Update waypoint landing"""
        if self.current_view_mode == "pointcloud":
            self.main_point_cloud.update_waypoint(index, landing=(state == Qt.Checked))
    
    def add_waypoint(self, index):
        """Add waypoint"""
        if self.current_view_mode == "pointcloud":
            self.main_point_cloud.update_waypoint(index, added=True)
            waypoints = self.main_point_cloud.get_waypoints()
            pos = waypoints[index]['position']
            self.log_debug(f"Waypoint {index+1} added: ({pos[0]:.2f}, {pos[1]:.2f})")
            self.send_waypoints_to_server()
    
    def delete_waypoint(self, index):
        """Delete waypoint"""
        if self.current_view_mode == "pointcloud":
            self.main_point_cloud.delete_waypoint(index)
            self.log_debug(f"Waypoint {index+1} deleted")
            self.send_waypoints_to_server()
    
    def send_multiple_commands(self):
        """Send multiple waypoints and start mission"""
        if self.current_view_mode != "pointcloud":
            self.log_debug("Cannot send waypoints: not in point cloud view")
            return
        
        try:
            waypoints = self.main_point_cloud.get_waypoints()
            added = [wp for wp in waypoints if wp.get('added', False)]
            
            if not added:
                self.log_debug("No waypoints to send")
                return
            
            # Format waypoints
            wp_strings = []
            for wp in added:
                x, y = wp['position']
                wp_str = f"[{x:.2f},{y:.2f},{wp['orientation']:.3f}," \
                         f"{int(wp['yaw_enable'])},{int(wp['landing'])}]"
                wp_strings.append(wp_str)
            
            # Send coordinates
            cmd = f"coordinates [{','.join(wp_strings)}]"
            if self.send_websocket_command(cmd):
                self.log_debug(f"Sent {len(wp_strings)} waypoints")
                time.sleep(0.1)
                
                # Start mission
                if self.send_websocket_command("start"):
                    self.log_debug("Mission started")
                else:
                    self.log_debug("Failed to start mission")
            else:
                self.log_debug("Failed to send waypoints")
        
        except Exception as e:
            self.log_debug(f"Error: {e}")
    
    def send_waypoints_to_server(self):
        """Send waypoints to server"""
        try:
            waypoints = self.main_point_cloud.get_waypoints()
            added = [wp for wp in waypoints if wp.get('added', False)]
            
            if not added:
                return False
            
            wp_strings = []
            for wp in added:
                x, y = wp['position']
                wp_str = f"[{x:.2f},{y:.2f},{wp['orientation']:.3f}," \
                         f"{int(wp['yaw_enable'])},{int(wp['landing'])}]"
                wp_strings.append(wp_str)
            
            cmd = f"coordinates [{','.join(wp_strings)}]"
            return self.send_websocket_command(cmd)
        
        except Exception as e:
            self.log_debug(f"Error sending waypoints: {e}")
            return False
    
    def save_current_frame(self):
        """Save current point cloud frame"""
        if self.current_view_mode != "pointcloud":
            self.log_debug("Cannot save: not in point cloud view")
            return
        
        raw_points = self.main_point_cloud.raw_points
        if len(raw_points) == 0:
            self.log_debug("No frame to save")
            return
        
        try:
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
        
        except Exception as e:
            self.log_debug(f"Save error: {e}")
    
    # ========================================
    # UTILITY METHODS
    # ========================================
    
    def on_orientation_dial_changed(self, value):
        """Handle orientation dial change"""
        import math
        orientation = (value / 50.0 - 1.0) * math.pi
        self.ui.mcOrientation.setText(f"{orientation:.4f} rad")
    
    def update_altitude_display(self, value=None):
        """Update altitude slider from DroneHeight label"""
        try:
            height_text = self.ui.DroneHeight.text().strip()
            height_text = height_text.replace(" meter", "").replace(" m", "").strip()
            altitude = float(height_text)
            slider_value = int(round(altitude * 100))
            self.ui.DroneAltitude.setValue(slider_value)
        except ValueError:
            pass
    
    def update_fps_display(self):
        """Update FPS display"""
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed > 0:
            main_fps = self.frame_counters['main'] / elapsed
            cam1_fps = self.frame_counters['cam1'] / elapsed
            cam2_fps = self.frame_counters['cam2'] / elapsed
            
            fps_text = f"FPS - Main: {main_fps:.1f} | Cam1: {cam1_fps:.1f} | Cam2: {cam2_fps:.1f}"
            self.log_debug(fps_text)
            
            self.frame_counters = {'main': 0, 'cam1': 0, 'cam2': 0}
            self.last_fps_time = current_time
    
    def log_debug(self, message):
        """Log message to debugging console"""
        timestamp = time.strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        
        self.ui.tbDebugging.append(formatted)
        
        # Keep only last 100 lines
        doc = self.ui.tbDebugging.document()
        if doc.blockCount() > 100:
            cursor = self.ui.tbDebugging.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
    
    # ========================================
    # CLEANUP
    # ========================================
    
    def closeEvent(self, event):
        """Clean shutdown"""
        print("\nShutting down Drone Control Center...")
        
        # Stop AI timer
        if hasattr(self, 'ai_timer'):
            self.ai_timer.stop()
        
        # Cleanup AI controller
        if self.ai_controller:
            try:
                self.ai_controller.cleanup()
                print("AI controller cleaned up")
            except Exception as e:
                print(f"AI cleanup error: {e}")
        
        # Stop video streams
        try:
            if hasattr(self, 'video_stream'):
                self.video_stream.stop_stream()
                time.sleep(0.5)
        except Exception as e:
            print(f"Video stop error: {e}")
        
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
                self.drone_parser.stop()
            except Exception as e:
                print(f"Drone parser stop error: {e}")
        
        # Save waypoints
        if hasattr(self, 'main_point_cloud'):
            try:
                self.main_point_cloud.save_waypoints_to_json()
                print("Waypoints saved")
            except Exception as e:
                print(f"Waypoint save error: {e}")
        
        print("System shutdown complete\n")
        event.accept()