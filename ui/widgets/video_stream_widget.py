# ui/widgets/video_stream_widget.py - ENHANCED MULTI-CAMERA VERSION
import os
import sys

# OpenCV import (cross-platform with fallback)
cv_folder = os.path.join(os.getcwd(), "cv_custom")
if os.path.exists(cv_folder):
    os.add_dll_directory(cv_folder)
    sys.path.insert(0, cv_folder)
    os.environ.setdefault("GST_PLUGIN_PATH", cv_folder)

try:
    import cv_custom as cv
    has_gstreamer = True
    has_ffmpeg = True
    print("✓ Using custom OpenCV build")
except Exception:
    import cv2 as cv
    has_gstreamer = False
    has_ffmpeg = False
    print("✓ Using standard OpenCV")

print("OpenCV version:", cv.__version__)

import threading
import time
import socket
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont

class RTSPConnectionTester:
    """Test RTSP connection using different methods"""

    @staticmethod
    def test_rtsp_connectivity(rtsp_url, timeout=5):
        """Test if RTSP stream is reachable"""
        try:
            if not rtsp_url or not isinstance(rtsp_url, str):
                return False, "Invalid URL type"
            
            rtsp_url = rtsp_url.strip()
            
            if not rtsp_url.startswith('rtsp://'):
                return False, f"Not an RTSP URL: {rtsp_url[:20]}"
            
            url_part = rtsp_url[7:]
            
            if '/' in url_part:
                host_port = url_part.split('/')[0]
            else:
                host_port = url_part
            
            if ':' in host_port:
                parts = host_port.rsplit(':', 1)
                host = parts[0]
                try:
                    port = int(parts[1])
                except (ValueError, IndexError):
                    port = 554
            else:
                host = host_port
                port = 554
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                return True, f"RTSP server reachable at {host}:{port}"
            else:
                return False, f"Cannot connect to {host}:{port}"
                
        except Exception as e:
            return False, f"Connection test failed: {e}"


class RTSPCamera(QThread):
    """Independent RTSP Camera thread - continues running even if other cameras fail"""
    
    frame_ready = pyqtSignal(np.ndarray)
    connection_status_changed = pyqtSignal(str)
    camera_failed = pyqtSignal(str)  # New signal for critical failures
    
    def __init__(self, rtsp_url, camera_id, ai_worker=None, width_scale=0.5, height_scale=0.5, latency_ms=50):
        super().__init__()
        self.rtsp_url = rtsp_url.strip() if rtsp_url else ""
        self.camera_id = camera_id
        self.ai_worker = ai_worker
        self.width_scale = width_scale
        self.height_scale = height_scale
        self.latency_ms = latency_ms
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.cap = None
        self.failed = False  # Track if camera has permanently failed
        
    def set_ai_worker(self, ai_worker):
        """Set AI worker untuk camera ini"""
        self.ai_worker = ai_worker
        print(f"🤖 AI worker set for camera {self.camera_id}")
    
    def run(self):
        """Main thread with independent error handling"""
        print(f"🚀 Starting RTSP camera {self.camera_id}: {self.rtsp_url[:50]}")
        
        if not self.rtsp_url:
            self.connection_status_changed.emit("Invalid URL")
            self.failed = True
            self.camera_failed.emit(f"Camera {self.camera_id}: Invalid URL")
            return
        
        self.running = True
        
        # Test connectivity
        self.connection_status_changed.emit("Testing Connection")
        reachable, msg = RTSPConnectionTester.test_rtsp_connectivity(self.rtsp_url)
        print(f"📡 Camera {self.camera_id} connectivity: {msg}")
        
        if not reachable:
            self.connection_status_changed.emit("Server Unreachable")
            self.running = False
            self.failed = True
            self.camera_failed.emit(f"Camera {self.camera_id}: {msg}")
            return
        
        # Try connection methods
        connection_methods = []
        
        if has_gstreamer:
            pipeline = (
                f"rtspsrc location={self.rtsp_url} latency={self.latency_ms} ! "
                "rtph264depay ! h264parse ! avdec_h264 ! "
                "videoconvert ! appsink"
            )
            connection_methods.append(("GStreamer", cv.CAP_GSTREAMER, pipeline))
        
        if has_ffmpeg:
            connection_methods.append(("FFMPEG", cv.CAP_FFMPEG, self.rtsp_url))
        
        connection_methods.append(("Default", cv.CAP_ANY, self.rtsp_url))
        
        # Try each method
        for method_name, backend, source in connection_methods:
            if not self.running:
                break
                
            self.connection_status_changed.emit(f"Trying {method_name}")
            print(f"🔧 Camera {self.camera_id} - {method_name}...")
            
            try:
                self.cap = cv.VideoCapture(source, backend)
                
                if backend == cv.CAP_FFMPEG:
                    self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv.CAP_PROP_FPS, 30)
                
                time.sleep(1)
                
                if self.cap.isOpened():
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        print(f"✅ Camera {self.camera_id} connected via {method_name}")
                        break
                    else:
                        self.cap.release()
                        self.cap = None
                else:
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                    
            except Exception as e:
                print(f"❌ Camera {self.camera_id} {method_name} error: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
        
        if not self.cap or not self.cap.isOpened():
            self.connection_status_changed.emit("All Methods Failed")
            print(f"❌ Camera {self.camera_id} - all methods failed")
            self.running = False
            self.failed = True
            self.camera_failed.emit(f"Camera {self.camera_id}: All connection methods failed")
            return
        
        self.connection_status_changed.emit("Connected")
        print(f"✅ Camera {self.camera_id} connected - entering main loop")
        
        frame_count = 0
        consecutive_failures = 0
        max_failures = 30
        
        # MAIN CAPTURE LOOP - Independent execution
        while self.running:
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print(f"❌ Camera {self.camera_id} - too many failures, stopping")
                        self.failed = True
                        self.camera_failed.emit(f"Camera {self.camera_id}: Frame read failures")
                        break
                    self.msleep(100)
                    continue
                
                consecutive_failures = 0
                frame_count += 1
                
                # Store frame
                with self.lock:
                    self.frame = frame.copy()
                
                # Emit raw frame (AI processing handled externally)
                self.frame_ready.emit(frame)
                
                # Small sleep to prevent CPU overload
                if frame_count % 100 == 0:
                    self.msleep(1)
                
            except Exception as e:
                print(f"❌ Camera {self.camera_id} capture error: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    self.failed = True
                    self.camera_failed.emit(f"Camera {self.camera_id}: Critical error - {str(e)}")
                    break
                self.msleep(100)
        
        # Cleanup
        print(f"🛑 Camera {self.camera_id} stopped ({frame_count} frames)")
        
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        
        self.running = False
        if not self.failed:
            self.connection_status_changed.emit("Disconnected")
        
    def stop(self):
        """Stop the camera thread safely"""
        print(f"🛑 Stopping camera {self.camera_id}")
        self.running = False
        
        if self.isRunning():
            self.wait(3000)
            
            if self.isRunning():
                print(f"⚠️ Camera {self.camera_id} terminating...")
                self.terminate()
                self.wait()
        
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        
        print(f"✅ Camera {self.camera_id} stopped")


class VideoStreamWidget(QWidget):
    """Widget for displaying video streaming."""
    
    frame_received = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 200)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: rgb(30, 30, 30);
                border: 1px solid rgb(60, 60, 60);
                color: rgb(150, 150, 150);
                font-size: 14px;
            }
        """)
        
        self.layout.addWidget(self.video_label)
        
        self.current_frame = None
        self.frame_count = 0
        self.is_connected = False
        
        self.show_info_overlay = True
        self.fps = 0.0
        self.connection_status = "Disconnected"
        
    def update_frame(self, frame_data):
        """Update with new video frame."""
        try:
            if isinstance(frame_data, np.ndarray):
                if len(frame_data.shape) == 3:
                    height, width, channels = frame_data.shape
                    if channels == 3:
                        rgb_frame = cv.cvtColor(frame_data, cv.COLOR_BGR2RGB)
                        qimage = QImage(rgb_frame.data, width, height, 
                                      width * 3, QImage.Format_RGB888)
                    else:
                        return
                elif len(frame_data.shape) == 2:
                    height, width = frame_data.shape
                    qimage = QImage(frame_data.data, width, height, 
                                  width, QImage.Format_Grayscale8)
                else:
                    return
                    
            elif isinstance(frame_data, bytes):
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
                if frame is not None:
                    height, width, channels = frame.shape
                    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    qimage = QImage(rgb_frame.data, width, height, 
                                  width * 3, QImage.Format_RGB888)
                else:
                    return
                    
            elif isinstance(frame_data, QImage):
                qimage = frame_data
            else:
                return
            
            self.current_frame = qimage
            self.frame_count += 1
            self.is_connected = True
            self.connection_status = "Connected"
            
            current_time = time.time()
            if hasattr(self, '_last_frame_time'):
                time_diff = current_time - self._last_frame_time
                if time_diff > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / time_diff)
            self._last_frame_time = current_time
            
            self.update_display()
            self.frame_received.emit()
            
        except Exception as e:
            print(f"Error updating video frame: {e}")
    
    def update_display(self):
        """Update video display with current frame."""
        if self.current_frame is None:
            return
            
        widget_size = self.video_label.size()
        
        scaled_image = self.current_frame.scaled(
            widget_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        if self.show_info_overlay:
            scaled_image = self.add_info_overlay(scaled_image)
        
        pixmap = QPixmap.fromImage(scaled_image)
        self.video_label.setPixmap(pixmap)
    
    def add_info_overlay(self, qimage):
        """Add information overlay to image."""
        overlay_image = qimage.copy()
        painter = QPainter(overlay_image)
        painter.setRenderHint(QPainter.Antialiasing)
        
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        
        overlay_text = f"FPS: {self.fps:.1f} | Frames: {self.frame_count}"
        
        lines = overlay_text.split('\n')
        line_height = painter.fontMetrics().height()
        max_width = max(painter.fontMetrics().width(line) for line in lines)
        
        padding = 5
        rect_width = max_width + 2 * padding
        rect_height = len(lines) * line_height + 2 * padding
        
        overlay_rect = painter.fontMetrics().boundingRect(10, 10, rect_width, rect_height, 0, "")
        painter.fillRect(overlay_rect, QColor(0, 0, 0, 128))
        
        y_pos = 10 + padding + line_height
        for line in lines:
            painter.drawText(10 + padding, y_pos, line)
            y_pos += line_height
        
        painter.end()
        return overlay_image
    
    def set_no_signal_message(self, message="No Video Stream"):
        """Set message when no video signal."""
        if not self.is_connected:
            self.video_label.setText(f"{message}\nWaiting for connection...")
    
    def clear_video(self):
        """Clear video display."""
        self.current_frame = None
        self.is_connected = False
        self.frame_count = 0
        self.fps = 0.0
        self.connection_status = "Disconnected"
        self.video_label.clear()
        self.video_label.setText("No Video Stream\nWaiting for connection...")
    
    def set_info_overlay(self, enabled):
        """Toggle info overlay display."""
        self.show_info_overlay = enabled
        if self.current_frame:
            self.update_display()
    
    def resizeEvent(self, event):
        """Handle widget resize."""
        super().resizeEvent(event)
        if self.current_frame:
            self.update_display()


class RTSPStreamWidget(VideoStreamWidget):
    """Multi-Camera video streaming widget with INDEPENDENT camera handling"""
    
    def __init__(self):
        super().__init__()
        self.cameras = {}
        self.camera_states = {}  # Track each camera's state
        self.ai_controller = None
        
    def set_ai_controller(self, ai_controller):
        """Set AI controller untuk multi-camera processing"""
        self.ai_controller = ai_controller
        print("🤖 AI Controller set for RTSPStreamWidget")
        
    def start_single_camera(self, cam_id, url, width_scale=0.5, height_scale=0.5, enable_ai=True):
        """Start single camera independently - other cameras unaffected"""
        try:
            print(f"\n📹 Starting {cam_id}...")
            
            # Get AI worker if available
            ai_worker = None
            if enable_ai and self.ai_controller:
                cam_index = int(cam_id.replace('cam', ''))
                if cam_index < len(self.ai_controller.workers):
                    ai_worker = self.ai_controller.workers[cam_index]
            
            # Create and start camera thread
            camera = RTSPCamera(
                url, 
                camera_id=cam_id,
                ai_worker=ai_worker,
                width_scale=width_scale, 
                height_scale=height_scale,
                latency_ms=0
            )
            
            # Connect signals
            camera.connection_status_changed.connect(
                lambda status, cid=cam_id: self.on_camera_status_changed(cid, status)
            )
            camera.camera_failed.connect(
                lambda msg: self.on_camera_failed(msg)
            )
            
            # Start camera
            camera.start()
            self.cameras[cam_id] = camera
            self.camera_states[cam_id] = "starting"
            
            print(f"   ✅ {cam_id} thread started")
            return True
            
        except Exception as e:
            print(f"   ❌ {cam_id} failed to start: {e}")
            self.camera_states[cam_id] = "failed"
            return False
    
    @pyqtSlot(str, str)
    def on_camera_status_changed(self, cam_id, status):
        """Handle camera status changes"""
        self.camera_states[cam_id] = status.lower()
        print(f"📡 {cam_id}: {status}")
    
    @pyqtSlot(str)
    def on_camera_failed(self, message):
        """Handle camera failure - does not affect other cameras"""
        print(f"⚠️ CAMERA FAILURE: {message}")
        # Other cameras continue operating normally
    
    def start_stream(self, rtsp_url=None, cam1_url=None, cam2_url=None, 
                     width_scale=0.5, height_scale=0.5, enable_ai=False):
        """
        Start all camera streams independently
        If one camera fails, others continue operating
        """
        if not rtsp_url:
            rtsp_url = "rtsp://192.168.1.99:1234"
        if not cam1_url:
            cam1_url = "rtsp://192.168.1.88:8555/bottom"
        if not cam2_url:
            cam2_url = "rtsp://192.168.1.88:8554/top"
    
        print(f"\n{'='*60}")
        print(f"🎬 STARTING INDEPENDENT MULTI-CAMERA STREAMS (AI: {enable_ai})")
        print(f"{'='*60}")
        print(f"Main Camera: {rtsp_url}")
        print(f"Camera 1 (Bottom): {cam1_url}")
        print(f"Camera 2 (Top): {cam2_url}")
        print(f"AI Controller: {'SET' if self.ai_controller else 'NOT SET'}")
        print(f"{'='*60}\n")
        
        success_count = 0
        
        # Start Camera 0 independently
        if self.start_single_camera("cam0", rtsp_url, width_scale, height_scale, enable_ai):
            if "cam0" in self.cameras:
                self.cameras["cam0"].frame_ready.connect(self.on_cam0_frame)
                success_count += 1
        time.sleep(0.3)
        
        # Start Camera 1 independently
        if self.start_single_camera("cam1", cam1_url, 0.4, 0.4, enable_ai):
            if "cam1" in self.cameras:
                self.cameras["cam1"].frame_ready.connect(self.on_cam1_frame)
                success_count += 1
        time.sleep(0.3)
        
        # Start Camera 2 independently
        if self.start_single_camera("cam2", cam2_url, 0.4, 0.4, enable_ai):
            if "cam2" in self.cameras:
                self.cameras["cam2"].frame_ready.connect(self.on_cam2_frame)
                success_count += 1
        time.sleep(1)
        
        # Status report
        print(f"\n{'='*60}")
        print("📊 CAMERA STATUS CHECK")
        print(f"{'='*60}")
        
        for cam_id in ["cam0", "cam1", "cam2"]:
            if cam_id in self.cameras:
                camera = self.cameras[cam_id]
                is_running = camera.isRunning()
                has_cap = camera.cap is not None
                cap_opened = camera.cap.isOpened() if has_cap else False
                failed = camera.failed
                
                if failed:
                    status = "❌ FAILED"
                elif is_running and cap_opened:
                    status = "✅ ACTIVE"
                else:
                    status = "⚠️ STARTING"
                
                print(f"{cam_id.upper():10} - {status}")
                print(f"           Thread: {is_running} | Capture: {cap_opened} | Failed: {failed}")
            else:
                print(f"{cam_id.upper():10} - ❌ NOT STARTED")
            print()
        
        print(f"{'='*60}")
        print(f"✅ {success_count}/3 cameras started successfully")
        print(f"{'='*60}\n")
        
        return success_count > 0  # Success if at least one camera started
    
    def on_cam0_frame(self, frame):
        """Store Camera 0 frame"""
        if not hasattr(self, 'cam0_latest_frame'):
            self.cam0_latest_frame = None
        self.cam0_latest_frame = frame.copy()
    
    def on_cam1_frame(self, frame):
        """Store Camera 1 frame"""
        if not hasattr(self, 'cam1_latest_frame'):
            self.cam1_latest_frame = None
        self.cam1_latest_frame = frame.copy()
    
    def on_cam2_frame(self, frame):
        """Store Camera 2 frame"""
        if not hasattr(self, 'cam2_latest_frame'):
            self.cam2_latest_frame = None
        self.cam2_latest_frame = frame.copy()
    
    def get_active_cameras(self):
        """Get list of currently active cameras"""
        active = []
        for cam_id, camera in self.cameras.items():
            if not camera.failed and camera.isRunning():
                active.append(cam_id)
        return active
    
    def stop_stream(self):
        """Stop all camera streams"""
        print("\n🛑 Stopping all camera streams...")
        for key, camera in self.cameras.items():
            print(f"  Stopping {key}...")
            try:
                camera.stop()
            except Exception as e:
                print(f"  Error stopping {key}: {e}")
        
        self.cameras.clear()
        self.camera_states.clear()
        self.clear_video()
        print("⏹ All streams stopped\n")


class TCPVideoStreamWidget(VideoStreamWidget):
    """TCP video streaming widget."""
    
    def __init__(self):
        super().__init__()
        self.tcp_receiver = None
    
    def set_tcp_receiver(self, tcp_receiver):
        """Set TCP receiver for video data."""
        self.tcp_receiver = tcp_receiver
        if hasattr(tcp_receiver, 'video_frame_received'):
            tcp_receiver.video_frame_received.connect(self.update_frame)
    
    def process_tcp_data(self, data):
        """Process raw TCP data as video frame."""
        try:
            self.update_frame(data)
        except Exception as e:
            print(f"Error processing TCP video data: {e}")