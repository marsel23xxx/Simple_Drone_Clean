"""
Video Stream Widget - Integrated Version
Widget untuk menampilkan video streaming dari drone dengan GStreamer support
"""

import cv2
import numpy as np
import threading
import time
import os
import sys
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont

# Setup untuk custom OpenCV (dari file teman)
def setup_custom_opencv():
    """Setup custom OpenCV dengan GStreamer support"""
    try:
        cv_folder = os.path.join(os.getcwd(), "cv_custom")
        if os.path.exists(cv_folder):
            os.add_dll_directory(cv_folder)
            sys.path.insert(0, cv_folder)
            os.environ["GST_PLUGIN_PATH"] = cv_folder
            
            # Optional: suppress GTK / pygobject warnings
            os.environ["GI_TYPELIB_PATH"] = ""
            os.environ["PYGOBJECT_WARNINGS"] = "0"
            
            import cv_custom as cv3
            return cv3
    except Exception as e:
        print(f"Failed to load custom OpenCV: {e}")
        
    return cv2  # Fallback to regular OpenCV


class RTSPCamera(QThread):
    """RTSP Camera class dengan GStreamer support (modified from teman's file)"""
    
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
        self.cv_module = setup_custom_opencv()
        
    def run(self):
        """Main thread untuk capture video"""
        self.running = True
        self.connection_status_changed.emit("Connecting...")
        
        # Try GStreamer pipeline first (from teman's implementation)
        gst_pipeline = (
            f'rtspsrc location={self.rtsp_url} latency=0 drop-on-latency=true ! '
            f'rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! '
            f'video/x-raw,format=BGR ! appsink drop=true sync=false max-buffers=1'
        )
        
        # Try custom OpenCV with GStreamer
        try:
            if hasattr(self.cv_module, 'CAP_GSTREAMER'):
                self.cap = self.cv_module.VideoCapture(gst_pipeline, self.cv_module.CAP_GSTREAMER)
            else:
                self.cap = self.cv_module.VideoCapture(gst_pipeline)
        except Exception as e:
            print(f"Failed to create GStreamer pipeline: {e}")
            # Fallback to direct RTSP
            self.cap = self.cv_module.VideoCapture(self.rtsp_url)

        if not self.cap.isOpened():
            print("❌ Failed to open RTSP stream")
            self.connection_status_changed.emit("Connection Failed")
            self.running = False
            return

        self.connection_status_changed.emit("Connected")
        print("✅ RTSP stream opened successfully")
        
        if hasattr(self.cv_module, 'getBuildInformation'):
            build_info = self.cv_module.getBuildInformation()
            gstreamer_enabled = "GStreamer:                   YES" in build_info
            print(f"OpenCV version: {self.cv_module.__version__}")
            print(f"GStreamer enabled: {gstreamer_enabled}")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Resize frame sesuai scale (dari implementasi teman)
            if self.width_scale != 1.0 or self.height_scale != 1.0:
                h, w = frame.shape[:2]
                frame = self.cv_module.resize(frame, 
                    (int(w * self.width_scale), int(h * self.height_scale)))

            with self.lock:
                self.frame = frame.copy()
                
            # Emit signal untuk update UI
            self.frame_ready.emit(frame)
            
            self.msleep(1)  # Small delay

    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()  # Wait for thread to finish

    def get_frame(self):
        """Get current frame (thread safe)"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None


class VideoStreamWidget(QWidget):
    """Widget untuk menampilkan video streaming dari drone."""
    
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
        self.video_label.setText("No Video Stream\nWaiting for connection...")
        
        self.layout.addWidget(self.video_label)
        
        # Current frame data
        self.current_frame = None
        self.frame_count = 0
        self.is_connected = False
        
        # Video properties
        self.video_width = 640
        self.video_height = 480
        
        # Info overlay
        self.show_info_overlay = True
        self.fps = 0.0
        self.last_frame_time = 0
        
        # Connection status
        self.connection_status = "Disconnected"
        
    def update_frame(self, frame_data):
        """Update dengan frame video baru.
        
        Args:
            frame_data: Bisa berupa:
                - numpy array (OpenCV format)
                - bytes (JPEG encoded)
                - QImage
        """
        try:
            # Convert frame_data to QImage
            if isinstance(frame_data, np.ndarray):
                # OpenCV format (BGR)
                if len(frame_data.shape) == 3:
                    height, width, channels = frame_data.shape
                    if channels == 3:
                        # Convert BGR to RGB
                        rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
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
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    height, width, channels = frame.shape
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        """Update video display dengan current frame."""
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
        # Create a copy to draw on
        overlay_image = qimage.copy()
        
        painter = QPainter(overlay_image)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set font and pen for overlay text
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        
        # Draw semi-transparent background for text
        overlay_rect = painter.fontMetrics().boundingRect(
            f"FPS: {self.fps:.1f} | Frames: {self.frame_count} | {self.connection_status}"
        )
        overlay_rect.adjust(-5, -2, 5, 2)
        overlay_rect.moveTopLeft(qimage.rect().topLeft())
        overlay_rect.translate(10, 10)
        
        painter.fillRect(overlay_rect, QColor(0, 0, 0, 128))
        
        # Draw text
        text_rect = overlay_rect.adjusted(5, 2, -5, -2)
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter,
                        f"FPS: {self.fps:.1f} | Frames: {self.frame_count} | {self.connection_status}")
        
        painter.end()
        return overlay_image
    
    def set_no_signal_message(self, message="No Video Stream"):
        """Set message ketika tidak ada sinyal video."""
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
    
    def get_video_info(self):
        """Get current video information."""
        return {
            'frame_count': self.frame_count,
            'fps': self.fps,
            'connected': self.is_connected,
            'resolution': (self.video_width, self.video_height) if self.current_frame else (0, 0)
        }
    
    def resizeEvent(self, event):
        """Handle widget resize."""
        super().resizeEvent(event)
        if self.current_frame:
            self.update_display()


class RTSPStreamWidget(VideoStreamWidget):
    """Video streaming widget dengan RTSP support menggunakan GStreamer."""
    
    def __init__(self, rtsp_url=None, width_scale=0.5, height_scale=0.5):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.width_scale = width_scale
        self.height_scale = height_scale
        self.rtsp_camera = None
        
    def start_stream(self, rtsp_url=None, width_scale=None, height_scale=None):
        """Start RTSP stream dengan GStreamer."""
        if rtsp_url:
            self.rtsp_url = rtsp_url
        if width_scale is not None:
            self.width_scale = width_scale
        if height_scale is not None:
            self.height_scale = height_scale
            
        if not self.rtsp_url:
            print("No RTSP URL provided")
            return False
            
        try:
            # Stop existing stream if running
            if self.rtsp_camera:
                self.stop_stream()
            
            # Create new RTSP camera instance
            self.rtsp_camera = RTSPCamera(
                self.rtsp_url, 
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
            print(f"Error starting RTSP stream: {e}")
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
        if status == "Connection Failed":
            self.clear_video()
            self.set_no_signal_message("Connection Failed")
        elif status == "Connecting...":
            self.set_no_signal_message("Connecting to RTSP stream...")
    
    def set_stream_quality(self, width_scale, height_scale):
        """Set stream quality scale."""
        self.width_scale = width_scale
        self.height_scale = height_scale
        
        if self.rtsp_camera:
            self.rtsp_camera.width_scale = width_scale
            self.rtsp_camera.height_scale = height_scale


class TCPVideoStreamWidget(VideoStreamWidget):
    """Video streaming widget dengan TCP socket support."""
    
    def __init__(self):
        super().__init__()
        self.tcp_receiver = None
    
    def set_tcp_receiver(self, tcp_receiver):
        """Set TCP receiver untuk video data."""
        self.tcp_receiver = tcp_receiver
        # Connect TCP receiver signal to update method
        if hasattr(tcp_receiver, 'video_frame_received'):
            tcp_receiver.video_frame_received.connect(self.update_frame)
    
    def process_tcp_data(self, data):
        """Process raw TCP data as video frame."""
        try:
            # Assume data is JPEG encoded
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
            self.setWindowTitle("RTSP Video Stream Demo")
            self.setGeometry(100, 100, 800, 600)
            
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            layout = QVBoxLayout(central_widget)
            
            # Video widget
            self.video_widget = RTSPStreamWidget()
            layout.addWidget(self.video_widget)
            
            # Control buttons
            button_layout = QHBoxLayout()
            
            self.start_button = QPushButton("Start Stream")
            self.start_button.clicked.connect(self.start_stream)
            button_layout.addWidget(self.start_button)
            
            self.stop_button = QPushButton("Stop Stream")
            self.stop_button.clicked.connect(self.stop_stream)
            button_layout.addWidget(self.stop_button)
            
            layout.addLayout(button_layout)
            
        def start_stream(self):
            # Ganti dengan URL RTSP Anda
            rtsp_url = "rtsp://192.168.1.99:1234"
            success = self.video_widget.start_stream(rtsp_url, width_scale=0.5, height_scale=0.5)
            if success:
                print("Stream started")
            else:
                print("Failed to start stream")
                
        def stop_stream(self):
            self.video_widget.stop_stream()
            print("Stream stopped")
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())