# video_stream_widget.py - FIXED VERSION
import os
import sys

# OpenCV import (cross-platform with fallback)
cv_folder = os.path.join(os.getcwd(), "cv_custom")

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
print("Available backends:", cv.getBuildInformation())

import threading
import time
import subprocess
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
            # Validasi input
            if not rtsp_url or not isinstance(rtsp_url, str):
                return False, "Invalid URL type"
            
            rtsp_url = rtsp_url.strip()
            
            if not rtsp_url.startswith('rtsp://'):
                return False, f"Not an RTSP URL: {rtsp_url[:20]}"
            
            # Parse URL - lebih robust
            url_part = rtsp_url[7:]  # hapus 'rtsp://'
            
            # Pisahkan host:port dari path
            if '/' in url_part:
                host_port = url_part.split('/')[0]
            else:
                host_port = url_part
            
            # Parse host dan port
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
            
            # Test koneksi socket
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
    
    @staticmethod
    def test_with_ffprobe(rtsp_url, timeout=10):
        """Test RTSP stream using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,codec_name',
                '-of', 'csv=p=0',
                '-timeout', str(timeout * 1000000),
                rtsp_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                return True, f"Stream info: {result.stdout.strip()}"
            else:
                return False, f"ffprobe failed: {result.stderr.strip()}"
                
        except subprocess.TimeoutExpired:
            return False, "ffprobe timeout"
        except FileNotFoundError:
            return False, "ffprobe not found (install FFmpeg)"
        except Exception as e:
            return False, f"ffprobe error: {e}"


class RTSPCamera(QThread):
    """RTSP Camera with multiple connection methods"""
    
    frame_ready = pyqtSignal(np.ndarray)
    connection_status_changed = pyqtSignal(str)
    
    def __init__(self, rtsp_url, width_scale=0.5, height_scale=0.5, latency_ms=50):
        super().__init__()
        self.rtsp_url = rtsp_url.strip() if rtsp_url else ""
        self.width_scale = width_scale
        self.height_scale = height_scale
        self.latency_ms = latency_ms
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.cap = None
        
        # Debug print
        print(f"📷 RTSPCamera initialized with URL: {self.rtsp_url}")
        
    def run(self):
        """Main thread for video capture with multiple fallback methods"""
        self.running = True
        self.connection_status_changed.emit("Testing Connection...")
        
        if cv is None:
            self.connection_status_changed.emit("OpenCV Not Available")
            self.running = False
            return
        
        # Test connectivity dengan URL yang sudah pasti ada
        print(f"🔍 Testing connectivity for: {self.rtsp_url}")
        is_reachable, msg = RTSPConnectionTester.test_rtsp_connectivity(self.rtsp_url)
        print(f"   Result: {msg}")
        
        if not is_reachable:
            self.connection_status_changed.emit("Server Unreachable")
            print(f"⚠️ Server unreachable, but will try to connect anyway...")
        
        connection_methods = [
            self._try_gstreamer_pipeline,
            self._try_ffmpeg_backend,
            self._try_direct_rtsp,
            self._try_direct_rtsp_with_options,
        ]
        
        success = False
        for i, method in enumerate(connection_methods):
            self.connection_status_changed.emit(f"Trying Method {i+1}/4...")
            
            # Timeout wrapper
            result = [False]
            
            def try_with_timeout():
                result[0] = method()
            
            thread = threading.Thread(target=try_with_timeout)
            thread.daemon = True
            thread.start()
            thread.join(timeout=15)  # 15 detik timeout
            
            if thread.is_alive():
                print(f"⏱️ Method {i+1} timeout after 15s")
                if self.cap:
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None
                continue
            
            if result[0]:
                success = True
                break
        
        if not success:
            print("❌ All connection methods failed")
            self.connection_status_changed.emit("All Methods Failed")
            self.running = False
            return
        
        self.connection_status_changed.emit("Connected")
        print("✅ RTSP stream connected successfully")
        
        frame_count = 0
        consecutive_failures = 0
        max_failures = 30
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print(f"❌ Too many consecutive failures ({consecutive_failures})")
                
                self.msleep(100)
                continue
            else:
                consecutive_failures = 0
            
            frame_count += 1
            if frame_count % 100 == 0:
                # print(f"📹 Frame count: {frame_count}")
                pass
            
            if self.width_scale != 1.0 or self.height_scale != 1.0:
                h, w = frame.shape[:2]
                frame = cv.resize(frame, 
                    (int(w * self.width_scale), int(h * self.height_scale)))
            
            with self.lock:
                self.frame = frame.copy()
            
            self.frame_ready.emit(frame)
            self.msleep(1)
        
        print("🛑 RTSP capture loop ended")
    
    def _try_gstreamer_pipeline(self):
        if not has_gstreamer:
            print("⏭️ Skipping GStreamer (not available)")
            return False
        
        try:
            print("🔄 Trying GStreamer pipeline...")
            
            gst_pipeline = (
                f'rtspsrc location={self.rtsp_url} latency={self.latency_ms} drop-on-latency=true protocols=tcp ! '
                f'rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! '
                f'video/x-raw,format=BGR ! appsink drop=true sync=false max-buffers=1'
            )
            self.cap = cv.VideoCapture(gst_pipeline, cv.CAP_GSTREAMER)
            
            if self.cap.isOpened():
                ret, test_frame = self.cap.read()
                if ret:
                    print("✅ GStreamer pipeline working")
                    return True
                else:
                    print("❌ GStreamer pipeline opened but no frames")
                    self.cap.release()
                    self.cap = None
            else:
                print("❌ GStreamer pipeline failed to open")
                
        except Exception as e:
            print(f"❌ GStreamer exception: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
        
        return False
    
    def _try_ffmpeg_backend(self):
        if not has_ffmpeg:
            print("⏭️ Skipping FFMPEG (not available)")
            return False
        
        try:
            print("🔄 Trying FFMPEG backend...")
            self.cap = cv.VideoCapture(self.rtsp_url, cv.CAP_FFMPEG)
            
            if self.cap.isOpened():
                self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    print("✅ FFMPEG backend working")
                    return True
                else:
                    print("❌ FFMPEG backend opened but no frames")
                    self.cap.release()
                    self.cap = None
            else:
                print("❌ FFMPEG backend failed to open")
                
        except Exception as e:
            print(f"❌ FFMPEG exception: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
        
        return False
    
    def _try_direct_rtsp(self):
        try:
            print(f"🔄 Trying direct RTSP for: {self.rtsp_url}")
            print(f"   URL length: {len(self.rtsp_url)}")
            print(f"   URL bytes: {self.rtsp_url.encode()}")
            
            # Set environment variables untuk RTSP
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
            
            self.cap = cv.VideoCapture(self.rtsp_url)
            print(f"   ✓ VideoCapture object created")
            print(f"   ✓ Is opened: {self.cap.isOpened()}")
            
            if self.cap.isOpened():
                # Set buffer minimal
                self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                
                print(f"   📖 Reading test frame (this may take 5-30 seconds)...")
                
                # Coba baca beberapa kali karena frame pertama sering gagal
                max_attempts = 5
                for attempt in range(max_attempts):
                    ret, test_frame = self.cap.read()
                    print(f"   Attempt {attempt+1}/{max_attempts}: ret={ret}, frame={'OK' if test_frame is not None else 'None'}")
                    
                    if ret and test_frame is not None:
                        h, w = test_frame.shape[:2]
                        print(f"✅ Direct RTSP working! Frame size: {w}x{h}")
                        return True
                    
                    # Tunggu sebentar sebelum coba lagi
                    time.sleep(0.5)
                
                print("❌ Direct RTSP opened but no frames after 5 attempts")
                self.cap.release()
                self.cap = None
            else:
                print("❌ Direct RTSP failed to open")
                
        except Exception as e:
            print(f"❌ Direct RTSP exception: {e}")
            import traceback
            traceback.print_exc()
            if self.cap:
                self.cap.release()
                self.cap = None
        
        return False
    
    def _try_direct_rtsp_with_options(self):
        try:
            print("🔄 Trying RTSP with TCP transport...")
            rtsp_tcp_url = self.rtsp_url
            if '?' in rtsp_tcp_url:
                rtsp_tcp_url += '&tcp'
            else:
                rtsp_tcp_url += '?tcp'
            
            self.cap = cv.VideoCapture(rtsp_tcp_url)
            
            if self.cap.isOpened():
                self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                
                print(f"   📖 Reading test frame with TCP...")
                max_attempts = 5
                for attempt in range(max_attempts):
                    ret, test_frame = self.cap.read()
                    print(f"   Attempt {attempt+1}/{max_attempts}: ret={ret}")
                    
                    if ret and test_frame is not None:
                        print("✅ RTSP with TCP transport working")
                        return True
                    
                    time.sleep(0.5)
                
                print("❌ RTSP TCP opened but no frames")
                self.cap.release()
                self.cap = None
            else:
                print("❌ RTSP TCP failed to open")
                
        except Exception as e:
            print(f"❌ RTSP TCP exception: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
        
        return False
    
    def stop(self):
        print("🛑 Stopping RTSP camera...")
        self.running = False
        if self.cap:
            self.cap.release()
    
    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None


class VideoStreamWidget(QWidget):
    """Widget for displaying video streaming from drone."""
    
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
        
        cv_info = "No OpenCV"
        if cv:
            gst_status = "GStreamer" if has_gstreamer else "No GStreamer"
            ffmpeg_status = "FFMPEG" if has_ffmpeg else "No FFMPEG"
            cv_info = f"OpenCV {cv.__version__} ({gst_status}, {ffmpeg_status})"
        
        # overlay_text = f"FPS: {self.fps:.1f} | Frames: {self.frame_count} | {self.connection_status}"
        overlay_text = f""
        # overlay_text += f"\n{cv_info}"
        
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
    """Video streaming widget with RTSP support - Like ARM_UI.py"""
    
    def __init__(self):
        super().__init__()
        # Camera objects dictionary
        self.cameras = {}
        self.main_camera_key = "rtsp"
        
    def start_stream(self, rtsp_url=None, cam1_url=None, cam2_url=None, 
                     width_scale=0.5, height_scale=0.5):
        """
        Start all camera streams
        Args:
            rtsp_url: Main RTSP camera URL
            cam1_url: Camera 1 (bottom) URL  
            cam2_url: Camera 2 (top) URL
            width_scale: Scale factor for width
            height_scale: Scale factor for height
        """
        if not rtsp_url:
            rtsp_url = "rtsp://192.168.1.99:1234"
        if not cam1_url:
            cam1_url = "rtsp://192.168.1.88:8555/bottom"
        if not cam2_url:
            cam2_url = "rtsp://192.168.1.88:8554/top"
    
        print(f"🎬 Starting RTSP streams...")
        print(f"  Main: {rtsp_url}")
        print(f"  Cam1: {cam1_url}")
        print(f"  Cam2: {cam2_url}")
    
        if cv is None:
            print("❌ OpenCV not available")
            self.set_no_signal_message("OpenCV Not Available")
            return False
    
        try:
            # Main RTSP camera - displays in main widget
            self.cameras["rtsp"] = RTSPCamera(
                rtsp_url, 
                width_scale=width_scale, 
                height_scale=height_scale,
                latency_ms=0
            )
            self.cameras["rtsp"].frame_ready.connect(self.update_frame)
            self.cameras["rtsp"].connection_status_changed.connect(self.update_connection_status)
            self.cameras["rtsp"].start()
            print(f"✓ Main RTSP camera started")
            
            # Camera 1 - Bottom (for AI processing)
            self.cameras["http1"] = RTSPCamera(
                cam1_url,
                width_scale=0.6,
                height_scale=0.6,
                latency_ms=0
            )
            self.cameras["http1"].frame_ready.connect(lambda frame: None)  # Silent
            self.cameras["http1"].start()
            print(f"✓ Camera 1 (Bottom) started")
            
            # Camera 2 - Top (for AI processing)
            self.cameras["http2"] = RTSPCamera(
                cam2_url,
                width_scale=0.6,
                height_scale=0.6,
                latency_ms=0
            )
            self.cameras["http2"].frame_ready.connect(lambda frame: None)  # Silent
            self.cameras["http2"].start()
            print(f"✓ Camera 2 (Top) started")
            
            return True
    
        except Exception as e:
            print(f"❌ Error starting streams: {e}")
            return False
    
    def get_all_frames(self):
        """Get frames from all cameras for AI processing"""
        frames = {}
        
        if "rtsp" in self.cameras:
            frame = self.cameras["rtsp"].get_frame()
            if frame is not None:
                frames['main'] = frame
        
        if "http1" in self.cameras:
            frame = self.cameras["http1"].get_frame()
            if frame is not None:
                frames['camera1'] = frame
        
        if "http2" in self.cameras:
            frame = self.cameras["http2"].get_frame()
            if frame is not None:
                frames['camera2'] = frame
        
        return frames
    
    def stop_stream(self):
        """Stop all camera streams properly"""
        for key, camera in self.cameras.items():
            print(f"Stopping {key} camera...")
            camera.stop()
        
        self.cameras.clear()
        self.clear_video()
        print("⏹ All streams stopped")
    
    @pyqtSlot(str)
    def update_connection_status(self, status):
        """Update connection status."""
        self.connection_status = status
        print(f"📡 Connection status: {status}")
        
        if status in ["All Methods Failed", "OpenCV Not Available", "Server Unreachable"]:
            self.clear_video()
            self.set_no_signal_message(status)
        elif "Trying" in status or "Testing" in status:
            self.set_no_signal_message(status)


class TCPVideoStreamWidget(VideoStreamWidget):
    """Video streaming widget with TCP socket support."""
    
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