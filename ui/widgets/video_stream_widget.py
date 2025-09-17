"""
Video Stream Widget
Widget untuk menampilkan video streaming dari drone
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont


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
            import time
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
    """Video streaming widget dengan RTSP support."""
    
    def __init__(self, rtsp_url=None):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.read_frame)
        
    def start_stream(self, rtsp_url=None):
        """Start RTSP stream."""
        if rtsp_url:
            self.rtsp_url = rtsp_url
            
        if not self.rtsp_url:
            print("No RTSP URL provided")
            return False
            
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if self.cap.isOpened():
                self.timer.start(33)  # ~30 FPS
                self.connection_status = "Connecting..."
                return True
            else:
                print(f"Failed to open RTSP stream: {self.rtsp_url}")
                return False
        except Exception as e:
            print(f"Error starting RTSP stream: {e}")
            return False
    
    def stop_stream(self):
        """Stop RTSP stream."""
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.clear_video()
    
    def read_frame(self):
        """Read frame from RTSP stream."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.update_frame(frame)
            else:
                self.connection_status = "Connection Lost"
                self.stop_stream()


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