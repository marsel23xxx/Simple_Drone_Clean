# video_stream_widget.py - System OpenCV with multiple RTSP methods
import os
import sys
import threading
import time
import subprocess
import socket
import numpy as np
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont

# Use system OpenCV only - avoid custom OpenCV completely
try:
    import cv2 as cv
    print(f"✅ System OpenCV loaded: {cv.__version__}")
    
    # Check capabilities
    if hasattr(cv, 'getBuildInformation'):
        build_info = cv.getBuildInformation()
        has_gstreamer = "GStreamer:                   YES" in build_info
        has_ffmpeg = "FFMPEG:                      YES" in build_info
        print(f"GStreamer support: {has_gstreamer}")
        print(f"FFMPEG support: {has_ffmpeg}")
    else:
        has_gstreamer = False
        has_ffmpeg = False
        print("Cannot check OpenCV build information")
        
except ImportError as e:
    print(f"❌ Failed to import system OpenCV: {e}")
    cv = None
    has_gstreamer = False
    has_ffmpeg = False


class RTSPConnectionTester:
    """Test RTSP connection using different methods"""
    
    @staticmethod
    def test_rtsp_connectivity(rtsp_url, timeout=5):
        """Test if RTSP stream is reachable"""
        try:
            # Parse RTSP URL to get host and port
            if rtsp_url.startswith('rtsp://'):
                url_part = rtsp_url[7:]  # Remove 'rtsp://'
                if ':' in url_part and '/' in url_part:
                    host_port = url_part.split('/')[0]
                    if ':' in host_port:
                        host, port = host_port.split(':')
                        port = int(port)
                    else:
                        host = host_port
                        port = 554  # Default RTSP port
                else:
                    return False, "Invalid RTSP URL format"
            else:
                return False, "Not an RTSP URL"
            
            # Test TCP connection
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
                '-timeout', str(timeout * 1000000),  # ffprobe timeout in microseconds
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
    
    def __init__(self, rtsp_url, width_scale=0.5, height_scale=0.5):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.width_scale = width_scale
        self.height_scale = height_scale
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.cap = None
        
    def run(self):
        """Main thread for video capture with multiple fallback methods"""
        self.running = True
        self.connection_status_changed.emit("Testing Connection...")
        
        if cv is None:
            self.connection_status_changed.emit("OpenCV Not Available")
            self.running = False
            return
        
        # Test connectivity first
        is_reachable, msg = RTSPConnectionTester.test_rtsp_connectivity(self.rtsp_url)
        print(f"RTSP connectivity test: {msg}")
        
        if not is_reachable:
            self.connection_status_changed.emit("Server Unreachable")
            # Don't return immediately - still try OpenCV methods
        
        # Try different connection methods in order of preference
        connection_methods = [
            self._try_gstreamer_pipeline,
            self._try_ffmpeg_backend,
            self._try_direct_rtsp,
            self._try_direct_rtsp_with_options,
        ]
        
        success = False
        for i, method in enumerate(connection_methods):
            self.connection_status_changed.emit(f"Trying Method {i+1}/4...")
            if method():
                success = True
                break
        
        if not success:
            print("❌ All connection methods failed")
            self.connection_status_changed.emit("All Methods Failed")
            self.running = False
            return
        
        self.connection_status_changed.emit("Connected")
        print("✅ RTSP stream connected successfully")
        
        # Main capture loop
        frame_count = 0
        consecutive_failures = 0
        max_failures = 30  # Allow up to 30 consecutive failed reads
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print(f"❌ Too many consecutive failures ({consecutive_failures})")
                    break
                self.msleep(100)  # Wait longer on failure
                continue
            else:
                consecutive_failures = 0  # Reset failure counter
            
            frame_count += 1
            if frame_count % 100 == 0:  # Log every 100 frames
                print(f"📹 Captured {frame_count} frames")
            
            # Resize frame if needed
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
        """Try GStreamer pipeline (if available)"""
        if not has_gstreamer:
            print("⏭️ Skipping GStreamer (not available)")
            return False
        
        try:
            print("🔄 Trying GStreamer pipeline...")
            gst_pipeline = (
                f'rtspsrc location={self.rtsp_url} latency=0 drop-on-latency=true ! '
                f'rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! '
                f'video/x-raw,format=BGR ! appsink drop=true sync=false max-buffers=1'
            )
            
            self.cap = cv.VideoCapture(gst_pipeline, cv.CAP_GSTREAMER)
            
            if self.cap.isOpened():
                # Test by reading a frame
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
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
        """Try FFMPEG backend"""
        if not has_ffmpeg:
            print("⏭️ Skipping FFMPEG (not available)")
            return False
        
        try:
            print("🔄 Trying FFMPEG backend...")
            self.cap = cv.VideoCapture(self.rtsp_url, cv.CAP_FFMPEG)
            
            if self.cap.isOpened():
                # Set some properties for better RTSP handling
                self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                
                # Test by reading a frame
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
        """Try direct RTSP without specifying backend"""
        try:
            print("🔄 Trying direct RTSP...")
            self.cap = cv.VideoCapture(self.rtsp_url)
            
            if self.cap.isOpened():
                # Test by reading a frame
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    print("✅ Direct RTSP working")
                    return True
                else:
                    print("❌ Direct RTSP opened but no frames")
                    self.cap.release()
                    self.cap = None
            else:
                print("❌ Direct RTSP failed to open")
                
        except Exception as e:
            print(f"❌ Direct RTSP exception: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
        
        return False
    
    def _try_direct_rtsp_with_options(self):
        """Try direct RTSP with modified URL options"""
        try:
            print("🔄 Trying RTSP with TCP transport...")
            # Force TCP transport (sometimes helps with connection issues)
            rtsp_tcp_url = self.rtsp_url
            if '?' in rtsp_tcp_url:
                rtsp_tcp_url += '&tcp'
            else:
                rtsp_tcp_url += '?tcp'
            
            self.cap = cv.VideoCapture(rtsp_tcp_url)
            
            if self.cap.isOpened():
                # Set minimal buffer
                self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                
                # Test by reading a frame
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    print("✅ RTSP with TCP transport working")
                    return True
                else:
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
        """Stop camera capture"""
        print("🛑 Stopping RTSP camera...")
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()
    
    def get_frame(self):
        """Get current frame (thread safe)"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None


class VideoStreamWidget(QWidget):
    """Widget for displaying video streaming from drone."""
    
    frame_received = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 200)
        
        # Setup layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Video display label
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
        # self.video_label.setText("No Video Stream\nWaiting for connection...")
        
        self.layout.addWidget(self.video_label)
        
        # Current frame data
        self.current_frame = None
        self.frame_count = 0
        self.is_connected = False
        
        # Info overlay
        self.show_info_overlay = True
        self.fps = 0.0
        
        # Connection status
        self.connection_status = "Disconnected"
        
    def update_frame(self, frame_data):
        """Update with new video frame."""
        try:
            # Convert frame_data to QImage
            if isinstance(frame_data, np.ndarray):
                # OpenCV format (BGR)
                if len(frame_data.shape) == 3:
                    height, width, channels = frame_data.shape
                    if channels == 3:
                        # Convert BGR to RGB
                        rgb_frame = cv.cvtColor(frame_data, cv.COLOR_BGR2RGB)
                        qimage = QImage(rgb_frame.data, width, height, 
                                      width * 3, QImage.Format_RGB888)
                    else:
                        return
                elif len(frame_data.shape) == 2:
                    # Grayscale
                    height, width = frame_data.shape
                    qimage = QImage(frame_data.data, width, height, 
                                  width, QImage.Format_Grayscale8)
                else:
                    return
                    
            elif isinstance(frame_data, bytes):
                # JPEG encoded data
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
            
            # Store current frame
            self.current_frame = qimage
            self.frame_count += 1
            self.is_connected = True
            self.connection_status = "Connected"
            
            # Calculate FPS
            current_time = time.time()
            if hasattr(self, '_last_frame_time'):
                time_diff = current_time - self._last_frame_time
                if time_diff > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / time_diff)
            self._last_frame_time = current_time
            
            # Update display
            self.update_display()
            self.frame_received.emit()
            
        except Exception as e:
            print(f"Error updating video frame: {e}")
    
    def update_display(self):
        """Update video display with current frame."""
        if self.current_frame is None:
            return
            
        # Get widget size
        widget_size = self.video_label.size()
        
        # Scale image to fit widget while maintaining aspect ratio
        scaled_image = self.current_frame.scaled(
            widget_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        # Add info overlay if enabled
        if self.show_info_overlay:
            scaled_image = self.add_info_overlay(scaled_image)
        
        # Convert to pixmap and display
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
        
        # Create overlay text with OpenCV info
        cv_info = "No OpenCV"
        if cv:
            gst_status = "GStreamer" if has_gstreamer else "No GStreamer"
            ffmpeg_status = "FFMPEG" if has_ffmpeg else "No FFMPEG"
            cv_info = f"OpenCV {cv.__version__} ({gst_status}, {ffmpeg_status})"
        
        overlay_text = f"FPS: {self.fps:.1f} | Frames: {self.frame_count} | {self.connection_status}"
        overlay_text += f"\n{cv_info}"
        
        # Calculate text area
        lines = overlay_text.split('\n')
        line_height = painter.fontMetrics().height()
        max_width = max(painter.fontMetrics().width(line) for line in lines)
        
        # Draw background rectangle
        padding = 5
        rect_width = max_width + 2 * padding
        rect_height = len(lines) * line_height + 2 * padding
        
        overlay_rect = painter.fontMetrics().boundingRect(10, 10, rect_width, rect_height, 0, "")
        painter.fillRect(overlay_rect, QColor(0, 0, 0, 128))
        
        # Draw text lines
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
    """Video streaming widget with RTSP support and multiple fallback methods."""
    
    def __init__(self, rtsp_url="rtsp://192.168.1.99:1234", base1="http://192.168.1.88:9001", base2="http://192.168.1.88:9002", width_scale=0.5, height_scale=0.5):
        super().__init__()
        self.base1= base1
        self.base2= base2
        self.rtsp_url = rtsp_url
        self.width_scale = width_scale
        self.height_scale = height_scale
        self.rtsp_camera = None
        self.start_stream(self.rtsp_url, self.base1, self.base2, self.width_scale, self.height_scale)
        
    def start_stream(self, rtsp_url="rtsp://192.168.1.99:1234", base1="http://192.168.1.88:9001", base2="http://192.168.1.88:9002", width_scale=None, height_scale=None):
        """Start RTSP stream with multiple fallback methods."""
        
        if rtsp_url:
            self.rtsp_url = rtsp_url
        if base1:
            self.base1 = base1
        if base2:
            self.base2 = base2
        if width_scale is not None:
            self.width_scale = width_scale
        if height_scale is not None:
            self.height_scale = height_scale
            
        if not self.rtsp_url:
            print("❌ No RTSP URL provided")
            return False
        if not self.base1:
            print("❌ No base1 URL provided")
            return False
        if not self.base2:
            print("❌ No base2 URL provided")
            return False
        
        print(f"🎬 Starting RTSP stream: {self.rtsp_url}")
        print(f"🎬 Starting Base1 stream: {self.base1}")
        print(f"🎬 Starting Base2 stream: {self.base2}")
        
        # Check if OpenCV is available
        if cv is None:
            print("❌ OpenCV not available")
            self.set_no_signal_message("OpenCV Not Available")
            return False
        
        try:
            # Stop existing stream if running
            if self.rtsp_camera:
                self.stop_stream()
            
            # Create new RTSP camera instance
            self.rtsp_camera = RTSPCamera(
                self.rtsp_url,
                self.base1,
                self.base2, 
                self.width_scale, 
                self.height_scale
            )
            
            # Connect signals
            self.rtsp_camera.frame_ready.connect(self.update_frame)
            self.rtsp_camera.connection_status_changed.connect(self.update_connection_status)
            
            # Start the camera thread
            self.rtsp_camera.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Error starting RTSP stream: {e}")
            return False
    
    def stop_stream(self):
        """Stop RTSP stream."""
        if self.rtsp_camera:
            self.rtsp_camera.stop()
            self.rtsp_camera = None
        self.clear_video()
    
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


# Demo usage
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
    import sys
    
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("RTSP Video Stream - System OpenCV Only")
            self.setGeometry(100, 100, 900, 700)
            
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            layout = QVBoxLayout(central_widget)
            
            # Video widget
            self.video_widget = RTSPStreamWidget()
            layout.addWidget(self.video_widget)
            
            # Control buttons
            button_layout = QHBoxLayout()
            
            self.start_stream()
            
            layout.addLayout(button_layout)
            
        def start_stream(self):
            rtsp_url = "rtsp://192.168.1.99:1234"
            base1 = "http://192.168.1.88:9001"
            base2 = "http://192.168.1.88:9002"
            success = self.video_widget.start_stream(rtsp_url, base1, base2, width_scale=0.5, height_scale=0.5)
            if success:
                print("✅ Stream start initiated")
            else:
                print("❌ Failed to start stream")
        
        def stop_stream(self):
            self.video_widget.stop_stream()
            print("🛑 Stream stopped")
        
        def test_connection(self):
            rtsp_url = "rtsp://192.168.1.99:1234"
            print("🔍 Testing RTSP connection...")
            
            # Test basic connectivity
            reachable, msg = RTSPConnectionTester.test_rtsp_connectivity(rtsp_url)
            print(f"Connectivity: {msg}")
            
            # Test with ffprobe if available
            stream_info, info_msg = RTSPConnectionTester.test_with_ffprobe(rtsp_url)
            print(f"Stream info: {info_msg}")
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())