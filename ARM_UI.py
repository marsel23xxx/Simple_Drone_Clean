#!/usr/bin/env python3
"""
Robotic Arm GUI Controller
Cross-platform: Works on Windows, Linux (ROS), and macOS
"""

import sys
import os


# OpenCV import (cross-platform with fallback)
cv_folder = "C:/Users/ADMIN/Documents/Simple_Drone_Clean/cv_custom"
print(cv_folder)

os.add_dll_directory(cv_folder)
sys.path.insert(0, cv_folder)
os.environ.setdefault("GST_PLUGIN_PATH", cv_folder)

try:
    import cv_custom as cv3
    print("✓ Using custom OpenCV build")
except Exception:
    import cv2 as cv3
    print("✓ Using standard OpenCV")
    
import threading
import time
import socket
import requests
import numpy as np
from typing import Optional
import asyncio
import websockets
# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QPushButton, QSlider, QGroupBox, QMessageBox, 
    QTextEdit, QInputDialog, QCheckBox
)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor, QTextCursor


# ========================= Helper Functions =========================

def get_local_ip():
    """Get the local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a port is open"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


# ========================= Camera Classes =========================

class RTSPCamera:
    """RTSP camera with GStreamer support"""
    
    def __init__(self, rtsp_url: str, width_scale: float = 0.5, height_scale: float = 0.5, latency_ms: int = 0):
        self.rtsp_url = rtsp_url
        self.width_scale = width_scale
        self.height_scale = height_scale
        self.latency_ms = latency_ms
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.cap = None
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    def _open(self):
        """Open RTSP stream with GStreamer or fallback"""
        # Try GStreamer pipeline first
        gst_pipeline = (
            f'rtspsrc location={self.rtsp_url} latency={self.latency_ms} drop-on-latency=true ! '
            f'rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! '
            f'video/x-raw,format=BGR ! appsink drop=true sync=false max-buffers=1'
        )
        
        cap = cv3.VideoCapture(gst_pipeline, cv3.CAP_GSTREAMER)
        
        if cap and cap.isOpened():
            print(f"✓ RTSP opened with GStreamer")
            return cap
        else:
            print("⚠ GStreamer failed, trying standard RTSP...")
            return cv3.VideoCapture(self.rtsp_url)

    def _loop(self):
        """Camera capture loop with auto-reconnect"""
        while self.running:
            self.cap = self._open()
            if not self.cap or not self.cap.isOpened():
                time.sleep(1)
                continue
                
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.02)
                    break
                    
                h, w = frame.shape[:2]
                if self.width_scale != 1.0 or self.height_scale != 1.0:
                    frame = cv3.resize(frame, (int(w * self.width_scale), int(h * self.height_scale)))
                    
                with self.lock:
                    self.frame = frame
                    
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


# ========================= WebSocket Client =========================

class WebSocketClient(QThread):
    """Thread-safe WebSocket client with auto-reconnect"""
    
    connected = pyqtSignal(bool)
    message_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self.ws = None
        self.running = False
        self.reconnect_delay = 3
        self.loop = None

    def connect_to_server(self):
        self.running = True
        self.start()

    def disconnect_from_server(self):
        self.running = False
        if self.isRunning():
            self.wait(2000)

    def send_message(self, message: str):
        if not self.ws or not self.running or not self.loop or self.loop.is_closed():
            self.error_occurred.emit("Not connected to server")
            return False
        
        try:
            future = asyncio.run_coroutine_threadsafe(self._send(message), self.loop)
            future.result(timeout=2.0)
            return True
        except asyncio.TimeoutError:
            self.error_occurred.emit("Send timeout - server not responding")
            return False
        except Exception as e:
            self.error_occurred.emit(f"Send error: {str(e)}")
            return False

    async def _send(self, message: str):
        if self.ws:
            try:
                await self.ws.send(message)
            except Exception as e:
                raise Exception(f"WebSocket send failed: {e}")
        else:
            raise Exception("WebSocket not connected")

    async def _connect_loop(self):
        """Main connection loop"""
        while self.running:
            try:
                async with websockets.connect(
                    self.url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                    open_timeout=10
                ) as websocket:
                    self.ws = websocket
                    self.connected.emit(True)
                    
                    try:
                        async for message in websocket:
                            if not self.running:
                                break
                            self.message_received.emit(message)
                    except websockets.exceptions.ConnectionClosedOK:
                        pass
                    except websockets.exceptions.ConnectionClosedError as e:
                        if self.running:
                            self.error_occurred.emit(f"Connection closed: {str(e)}")
                    
            except asyncio.TimeoutError:
                if self.running:
                    self.error_occurred.emit(f"Connection timeout")
                    self.connected.emit(False)
            except ConnectionRefusedError:
                if self.running:
                    self.error_occurred.emit(f"Connection refused - is server running?")
                    self.connected.emit(False)
            except Exception as e:
                if self.running:
                    self.error_occurred.emit(f"Connection failed: {str(e)}")
                    self.connected.emit(False)
            
            self.ws = None
            if self.running:
                self.connected.emit(False)
                await asyncio.sleep(self.reconnect_delay)
            else:
                break

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._connect_loop())
        except Exception as e:
            self.error_occurred.emit(f"Loop error: {str(e)}")
        finally:
            try:
                pending = asyncio.all_tasks(self.loop)
                for task in pending:
                    task.cancel()
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self.loop.close()
            except Exception:
                pass
            self.loop = None


# ========================= Custom Widgets =========================

class CameraDisplayWidget(QLabel):
    """Widget for displaying camera feed"""
    
    def __init__(self, title: str, width: int = 320, height: int = 240):
        super().__init__()
        self.title = title
        self.setMinimumSize(width, height)
        self.setStyleSheet("border: 2px solid #444; background-color: black;")
        self.setAlignment(Qt.AlignCenter)
        self.setText(f"{title}\n(No Feed)")

    def update_frame(self, frame):
        if frame is not None:
            rgb_image = cv3.cvtColor(frame, cv3.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)
        else:
            self.setText(f"{self.title}\n(No Feed)")


class ServoControlWidget(QWidget):
    """Individual servo control widget"""
    
    value_changed = pyqtSignal(int, int)

    def __init__(self, servo_number: int, min_val: int = 0, max_val: int = 180):
        super().__init__()
        self.servo_number = servo_number
        self.min_val = min_val
        self.max_val = max_val
        self.current_value = (min_val + max_val) // 2
        self.continuous_timer = QTimer()
        self.continuous_timer.timeout.connect(self.continuous_move)
        self.step_direction = 0
        self.step_size = 1
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        servo_label = QLabel(f"Servo {self.servo_number}")
        servo_label.setAlignment(Qt.AlignCenter)
        servo_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(servo_label)
        
        self.value_label = QLabel(str(self.current_value))
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet("font-weight: bold; font-size: 18px; color: #42a5f5;")
        layout.addWidget(self.value_label)
        
        self.up_btn = QPushButton("▲")
        self.up_btn.setMaximumHeight(35)
        self.up_btn.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.up_btn.pressed.connect(lambda: self.start_continuous_move(1))
        self.up_btn.released.connect(self.stop_continuous_move)
        layout.addWidget(self.up_btn)
        
        self.max_label = QLabel(str(self.max_val))
        self.max_label.setAlignment(Qt.AlignCenter)
        self.max_label.setStyleSheet("font-size: 12px; color: #888;")
        layout.addWidget(self.max_label)
        
        self.slider = QSlider(Qt.Vertical)
        self.slider.setMinimum(self.min_val)
        self.slider.setMaximum(self.max_val)
        self.slider.setValue(self.current_value)
        self.slider.valueChanged.connect(self.on_slider_change)
        self.slider.setMinimumHeight(300)
        layout.addWidget(self.slider)
        
        self.min_label = QLabel(str(self.min_val))
        self.min_label.setAlignment(Qt.AlignCenter)
        self.min_label.setStyleSheet("font-size: 12px; color: #888;")
        layout.addWidget(self.min_label)
        
        self.down_btn = QPushButton("▼")
        self.down_btn.setMaximumHeight(35)
        self.down_btn.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.down_btn.pressed.connect(lambda: self.start_continuous_move(-1))
        self.down_btn.released.connect(self.stop_continuous_move)
        layout.addWidget(self.down_btn)
        
        self.setLayout(layout)

    def start_continuous_move(self, direction):
        self.step_direction = direction
        self.continuous_timer.start(50)

    def stop_continuous_move(self):
        self.continuous_timer.stop()
        self.step_direction = 0

    def continuous_move(self):
        new_value = self.current_value + (self.step_direction * self.step_size)
        self.set_value(new_value)

    def on_slider_change(self, value):
        self.set_value(value, from_slider=True)

    def set_value(self, value, from_slider=False, emit_signal=True):
        value = max(self.min_val, min(self.max_val, value))
        if value != self.current_value:
            self.current_value = value
            if not from_slider:
                self.slider.setValue(value)
            self.value_label.setText(str(value))
            if emit_signal:
                self.value_changed.emit(self.servo_number, value)


# ========================= Main Application =========================

class RoboticArmControlGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robotic Arm Control System")
        self.setGeometry(50, 50, 1400, 900)
        
        # Get local IP
        local_ip = get_local_ip()
        
        # WebSocket URL - editable
        self.websocket_url = f"ws://{local_ip}:8080"
        
        # Camera objects
        self.cameras = {}
        self.main_camera_key = "rtsp"
        
        # WebSocket client (will be created on connect)
        self.ws_client = None
        
        # Servo controls
        self.servo_controls = {}
        
        # Preset poses
        self.poses = {
            "Home": [180, 180, 60, 90],
            "Forward": [180, 65, 180, 90],
            "Up": [180, 65, 135, 90],
            "Down": [180, 180, 105, 90]
        }
        
        self.init_ui()
        self.log_message(f"Local IP: {local_ip}")
        self.init_cameras()
        self.setup_update_timer()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        self.init_camera_panel(main_layout)
        self.init_control_panel(main_layout)

    def init_camera_panel(self, main_layout):
        camera_panel = QWidget()
        camera_layout = QVBoxLayout()
        camera_panel.setLayout(camera_layout)
        
        self.main_camera_display = CameraDisplayWidget("Main Camera", 640, 480)
        camera_layout.addWidget(self.main_camera_display)
        
        bottom_layout = QHBoxLayout()
        self.camera1_thumb = CameraDisplayWidget("Camera 1", 320, 240)
        self.camera2_thumb = CameraDisplayWidget("Camera 2", 320, 240)
        
        bottom_layout.addWidget(self.camera1_thumb)
        bottom_layout.addWidget(self.camera2_thumb)
        
        button_container = QWidget()
        button_layout = QVBoxLayout()
        button_container.setLayout(button_layout)
        
        self.switch_rtsp_btn = QPushButton("Main RTSP")
        self.switch_cam1_btn = QPushButton("Camera 1")
        self.switch_cam2_btn = QPushButton("Camera 2")
        
        self.switch_rtsp_btn.clicked.connect(lambda: self.switch_main_camera("rtsp"))
        self.switch_cam1_btn.clicked.connect(lambda: self.switch_main_camera("http1"))
        self.switch_cam2_btn.clicked.connect(lambda: self.switch_main_camera("http2"))
        
        for btn in [self.switch_rtsp_btn, self.switch_cam1_btn, self.switch_cam2_btn]:
            btn.setMinimumHeight(40)
        
        self.camera_buttons = {
            "rtsp": self.switch_rtsp_btn,
            "http1": self.switch_cam1_btn,
            "http2": self.switch_cam2_btn
        }
        
        button_layout.addWidget(self.switch_rtsp_btn)
        button_layout.addWidget(self.switch_cam1_btn)
        button_layout.addWidget(self.switch_cam2_btn)
        
        # Camera enable/disable checkboxes
        checkbox_container = QWidget()
        checkbox_layout = QVBoxLayout()
        checkbox_layout.setSpacing(8)
        checkbox_container.setLayout(checkbox_layout)
        
        checkbox_label = QLabel("Enable Cameras:")
        checkbox_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        checkbox_layout.addWidget(checkbox_label)
        
        self.rtsp_checkbox = QCheckBox("RTSP Camera")
        self.cam1_checkbox = QCheckBox("Camera 1")
        self.cam2_checkbox = QCheckBox("Camera 2")
        
        # Default to unchecked (off)
        self.rtsp_checkbox.setChecked(False)
        self.cam1_checkbox.setChecked(False)
        self.cam2_checkbox.setChecked(False)
        
        self.rtsp_checkbox.stateChanged.connect(lambda state: self.toggle_camera("rtsp", state))
        self.cam1_checkbox.stateChanged.connect(lambda state: self.toggle_camera("http1", state))
        self.cam2_checkbox.stateChanged.connect(lambda state: self.toggle_camera("http2", state))
        
        checkbox_layout.addWidget(self.rtsp_checkbox)
        checkbox_layout.addWidget(self.cam1_checkbox)
        checkbox_layout.addWidget(self.cam2_checkbox)
        
        button_layout.addWidget(checkbox_container)
        button_layout.addStretch()
        
        bottom_layout.addWidget(button_container)
        camera_layout.addLayout(bottom_layout)
        main_layout.addWidget(camera_panel, stretch=3)
        self.highlight_active_camera()

    def init_control_panel(self, main_layout):
        control_panel = QWidget()
        control_panel.setMaximumWidth(500)
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        
        self.init_connection_controls(control_layout)
        self.init_pose_controls(control_layout)
        self.init_servo_controls(control_layout)
        self.init_debug_log(control_layout)
        
        main_layout.addWidget(control_panel, stretch=1)

    def init_connection_controls(self, parent_layout):
        conn_group = QGroupBox("Connection")
        conn_layout = QVBoxLayout()
        
        url_layout = QHBoxLayout()
        url_label = QLabel("WebSocket:")
        self.url_display = QLabel(self.websocket_url)
        self.url_display.setStyleSheet("color: #aaa; font-family: monospace; font-size: 10px;")
        self.url_display.setWordWrap(True)
        
        edit_url_btn = QPushButton("Edit")
        edit_url_btn.setMaximumWidth(50)
        edit_url_btn.clicked.connect(self.edit_websocket_url)
        
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.url_display, stretch=1)
        url_layout.addWidget(edit_url_btn)
        
        status_layout = QHBoxLayout()
        self.connection_status = QLabel("●")
        self.connection_status.setStyleSheet("color: red; font-size: 20px;")
        
        self.connect_btn = QPushButton("Connect")
        self.disconnect_btn = QPushButton("Disconnect")
        
        self.connect_btn.clicked.connect(self.connect_websocket)
        self.disconnect_btn.clicked.connect(self.disconnect_websocket)
        self.disconnect_btn.setEnabled(False)
        
        status_layout.addWidget(self.connection_status)
        status_layout.addWidget(self.connect_btn)
        status_layout.addWidget(self.disconnect_btn)
        
        conn_layout.addLayout(url_layout)
        conn_layout.addLayout(status_layout)
        
        conn_group.setLayout(conn_layout)
        parent_layout.addWidget(conn_group)

    def init_pose_controls(self, parent_layout):
        pose_group = QGroupBox("Preset Poses")
        pose_layout = QGridLayout()
        
        row, col = 0, 0
        for pose_name in self.poses.keys():
            btn = QPushButton(pose_name)
            btn.clicked.connect(lambda checked, name=pose_name: self.execute_pose(name))
            btn.setMinimumHeight(50)
            btn.setStyleSheet("font-size: 14px; font-weight: bold;")
            pose_layout.addWidget(btn, row, col)
            col += 1
            if col >= 2:
                col = 0
                row += 1
        
        pose_group.setLayout(pose_layout)
        parent_layout.addWidget(pose_group)

    def init_servo_controls(self, parent_layout):
        servo_group = QGroupBox("Servo Controls")
        servo_layout = QHBoxLayout()
        
        for i in range(1, 5):
            servo_control = ServoControlWidget(i, 0, 180)
            servo_control.value_changed.connect(self.on_servo_value_changed)
            self.servo_controls[i] = servo_control
            servo_layout.addWidget(servo_control)
        
        servo_group.setLayout(servo_layout)
        parent_layout.addWidget(servo_group)

    def init_debug_log(self, parent_layout):
        debug_group = QGroupBox("Debug Log")
        debug_layout = QVBoxLayout()
        
        self.debug_log = QTextEdit()
        self.debug_log.setReadOnly(True)
        self.debug_log.setMaximumHeight(150)
        self.debug_log.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #00ff00;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10px;
                border: 1px solid #333;
            }
        """)
        
        debug_layout.addWidget(self.debug_log)
        
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.clear_debug_log)
        clear_btn.setMaximumHeight(25)
        debug_layout.addWidget(clear_btn)
        
        debug_group.setLayout(debug_layout)
        parent_layout.addWidget(debug_group)
        
        self.log_message("System initialized")

    def init_cameras(self):
        """Initialize cameras with error handling - but don't start them"""
        # Main RTSP camera
        rtsp_url = os.environ.get("RTSP_URL", "rtsp://192.168.1.99:1234")
        try:
            self.cameras["rtsp"] = RTSPCamera(rtsp_url, width_scale=0.8, height_scale=0.8)
            self.log_message(f"✓ RTSP camera created: {rtsp_url} (disabled)")
        except Exception as e:
            self.log_message(f"✗ RTSP camera failed: {e}")
        
        # RTSP cameras - bottom and top
        host = os.environ.get("CAMERA_HOST", "192.168.1.88")
        
        # Camera 1 - Bottom
        try:
            bottom_url = f"rtsp://{host}:8555/bottom"
            self.cameras["http1"] = RTSPCamera(bottom_url, width_scale=0.6, height_scale=0.6, latency_ms=0)
            self.log_message(f"✓ Camera 1 (Bottom) created: {bottom_url} (disabled)")
        except Exception as e:
            self.log_message(f"✗ Camera 1 failed: {e}")
        
        # Camera 2 - Top
        try:
            top_url = f"rtsp://{host}:8554/top"
            self.cameras["http2"] = RTSPCamera(top_url, width_scale=0.6, height_scale=0.6, latency_ms=0)
            self.log_message(f"✓ Camera 2 (Top) created: {top_url} (disabled)")
        except Exception as e:
            self.log_message(f"✗ Camera 2 failed: {e}")

    def setup_update_timer(self):
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_camera_displays)
        self.update_timer.start(33)

    def update_camera_displays(self):
        frames = {
            "rtsp": self.cameras.get("rtsp", {}).get_frame() if "rtsp" in self.cameras else None,
            "http1": self.cameras.get("http1", {}).get_frame() if "http1" in self.cameras else None,
            "http2": self.cameras.get("http2", {}).get_frame() if "http2" in self.cameras else None
        }
        
        if self.main_camera_key in frames:
            self.main_camera_display.update_frame(frames[self.main_camera_key])
        
        thumbnail_cameras = [k for k in ["rtsp", "http1", "http2"] if k != self.main_camera_key]
        
        if len(thumbnail_cameras) >= 1:
            self.camera1_thumb.update_frame(frames[thumbnail_cameras[0]])
        if len(thumbnail_cameras) >= 2:
            self.camera2_thumb.update_frame(frames[thumbnail_cameras[1]])

    def switch_main_camera(self, camera_key):
        if camera_key in self.cameras:
            self.main_camera_key = camera_key
            self.log_message(f"Switched to {self.get_camera_name(camera_key)}")
            self.highlight_active_camera()

    def toggle_camera(self, camera_key, state):
        """Enable or disable a camera based on checkbox state"""
        if camera_key not in self.cameras:
            self.log_message(f"✗ Camera {camera_key} not found")
            return
        
        camera = self.cameras[camera_key]
        
        if state == Qt.Checked:
            # Start camera
            try:
                camera.start()
                self.log_message(f"✓ {self.get_camera_name(camera_key)} enabled")
            except Exception as e:
                self.log_message(f"✗ Failed to start {self.get_camera_name(camera_key)}: {e}")
        else:
            # Stop camera
            try:
                camera.stop()
                self.log_message(f"○ {self.get_camera_name(camera_key)} disabled")
            except Exception as e:
                self.log_message(f"✗ Failed to stop {self.get_camera_name(camera_key)}: {e}")

    def get_camera_name(self, camera_key):
        names = {"rtsp": "Main RTSP", "http1": "Camera 1", "http2": "Camera 2"}
        return names.get(camera_key, camera_key)

    def highlight_active_camera(self):
        for key, button in self.camera_buttons.items():
            if key == self.main_camera_key:
                button.setStyleSheet("""
                    QPushButton {
                        background-color: #42a5f5;
                        color: white;
                        font-weight: bold;
                        border: 2px solid #1976d2;
                    }
                """)
            else:
                button.setStyleSheet("")

    def edit_websocket_url(self):
        new_url, ok = QInputDialog.getText(
            self, "Edit WebSocket URL",
            "Enter WebSocket URL (e.g., ws://192.168.1.110:8080):",
            text=self.websocket_url
        )
        
        if ok and new_url:
            if not new_url.startswith("ws://"):
                QMessageBox.warning(self, "Invalid URL", "URL must start with ws://")
                return
            
            self.websocket_url = new_url
            self.url_display.setText(new_url)
            
            if self.ws_client and self.ws_client.running:
                self.disconnect_websocket()
            
            self.log_message(f"URL updated: {new_url}")

    def connect_websocket(self):
        url = self.websocket_url
        
        try:
            host_port = url.replace("ws://", "")
            if ":" in host_port:
                host, port_str = host_port.split(":")
                port = int(port_str.split("/")[0])
            else:
                host = host_port.split("/")[0]
                port = 80
        except Exception as e:
            self.log_message(f"✗ Invalid URL: {e}")
            QMessageBox.warning(self, "Invalid URL", f"Cannot parse: {url}")
            return
        
        self.log_message(f"Connecting to {host}:{port}...")
        
        # Always disconnect and cleanup old client first
        if self.ws_client is not None:
            self.log_message("Cleaning up previous connection...")
            if self.ws_client.isRunning():
                self.ws_client.disconnect_from_server()
                self.ws_client.wait(2000)  # Wait up to 2 seconds for cleanup
            
            # Disconnect all signals to prevent ghost signals
            try:
                self.ws_client.connected.disconnect()
                self.ws_client.message_received.disconnect()
                self.ws_client.error_occurred.disconnect()
            except Exception:
                pass
            
            self.ws_client = None
        
        # Always create a NEW WebSocketClient instance (can't restart QThread)
        self.ws_client = WebSocketClient(url)
        self.ws_client.connected.connect(self.on_ws_connected)
        self.ws_client.message_received.connect(self.on_ws_message)
        self.ws_client.error_occurred.connect(self.on_ws_error)
        
        self.ws_client.connect_to_server()
        self.connect_btn.setEnabled(False)
        self.log_message("Connection initiated...")

    def disconnect_websocket(self):
        if self.ws_client:
            self.log_message("Disconnecting...")
            self.ws_client.disconnect_from_server()
            
            # Wait for thread to finish
            if self.ws_client.isRunning():
                self.ws_client.wait(3000)  # Wait up to 3 seconds
            
            self.log_message("Disconnected")
        
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.connection_status.setStyleSheet("color: red; font-size: 20px;")

    def on_ws_connected(self, connected):
        if connected:
            self.connection_status.setStyleSheet("color: lime; font-size: 20px;")
            self.disconnect_btn.setEnabled(True)
            self.log_message("✓ WebSocket connected")
        else:
            self.connection_status.setStyleSheet("color: red; font-size: 20px;")
            self.connect_btn.setEnabled(True)
            self.disconnect_btn.setEnabled(False)

    def on_ws_message(self, message: str):
        self.log_message(f"← {message}")

    def on_ws_error(self, error_msg):
        self.log_message(f"ERROR: {error_msg}")

    def execute_pose(self, pose_name):
        if pose_name in self.poses:
            values = self.poses[pose_name]
            
            for i, value in enumerate(values, start=1):
                if i in self.servo_controls:
                    self.servo_controls[i].set_value(value, emit_signal=False)
            
            command = f"#A {values[0]} {values[1]} {values[2]} {values[3]}"
            self.send_command(command)
            self.log_message(f"Pose '{pose_name}': {command}")

    def on_servo_value_changed(self, servo_number, value):
        command = f"#S {servo_number} {value}"
        self.send_command(command)

    def send_command(self, command):
        if self.ws_client and self.ws_client.send_message(command):
            self.log_message(f"→ {command}")
        else:
            self.log_message(f"✗ Failed: {command}")

    def log_message(self, message):
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.debug_log.append(f"[{timestamp}] {message}")
        self.debug_log.moveCursor(QTextCursor.End)

    def clear_debug_log(self):
        self.debug_log.clear()
        self.log_message("Log cleared")

    def closeEvent(self, event):
        for camera in self.cameras.values():
            camera.stop()
        
        if self.ws_client:
            self.ws_client.disconnect_from_server()
        
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Dark theme
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)
    
    window = RoboticArmControlGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()