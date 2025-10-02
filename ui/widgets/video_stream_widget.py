# video_stream_widget.py - FIXED: Main camera only in SwitchView
import os
import sys
import cv2 as cv

import threading
import time
import subprocess
import socket
import numpy as np
import requests
from urllib.parse import urlparse
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont

has_gstreamer = False
has_ffmpeg = True


class RTSPConnectionTester:
    """Test RTSP connection using different methods"""

    @staticmethod
    def test_rtsp_connectivity(rtsp_url, timeout=5):
        """Test if RTSP stream is reachable"""
        try:
            if rtsp_url.startswith('rtsp://'):
                url_part = rtsp_url[7:]
                if ':' in url_part and '/' in url_part:
                    host_port = url_part.split('/')[0]
                    if ':' in host_port:
                        host, port = host_port.split(':')
                        port = int(port)
                    else:
                        host = host_port
                        port = 554
                else:
                    return False, "Invalid RTSP URL format"
            else:
                return False, "Not an RTSP URL"
            
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
    
    def __init__(self, rtsp_url, base1, base2, width_scale=0.5, height_scale=0.5):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.base1 = base1
        self.base2 = base2
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
        
        is_reachable, msg = RTSPConnectionTester.test_rtsp_connectivity(self.rtsp_url)
        print(f"RTSP connectivity test: {msg}")
        
        if not is_reachable:
            self.connection_status_changed.emit("Server Unreachable")
        
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
        
        frame_count = 0
        consecutive_failures = 0
        max_failures = 30
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print(f"❌ Too many consecutive failures ({consecutive_failures})")
                    break
                self.msleep(100)
                continue
            else:
                consecutive_failures = 0
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"📹 Captured {frame_count} frames")
            
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
                f'rtspsrc location={self.rtsp_url} latency=0 drop-on-latency=true ! '
                f'rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! '
                f'video/x-raw,format=BGR ! appsink drop=true sync=false max-buffers=1'
            )
            
            self.cap = cv.VideoCapture(gst_pipeline, cv.CAP_GSTREAMER)
            
            if self.cap.isOpened():
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
            print("🔄 Trying direct RTSP...")
            self.cap = cv.VideoCapture(self.rtsp_url)
            
            if self.cap.isOpened():
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
        print("🛑 Stopping RTSP camera...")
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()
    
    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None


class HTTPStreamCamera:
    """Robust MJPEG reader for Flask/OpenCV endpoints"""
    def __init__(self, base_url: str, path: str = "/video_feed",
                 width_scale: float = 0.5, height_scale: float = 0.5):
        self.base_url = base_url.rstrip('/')
        self.url = self.base_url + (path if path.startswith('/') else '/' + path)
        self.width_scale = width_scale
        self.height_scale = height_scale
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop_mjpeg_manual, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _loop_mjpeg_manual(self):
        session = requests.Session()
        headers = {
            "Accept": "multipart/x-mixed-replace, image/jpeg",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Pragma": "no-cache",
        }
        while self.running:
            try:
                with session.get(self.url, stream=True, timeout=(3.0, 10.0), headers=headers) as r:
                    ctype = r.headers.get('Content-Type', '')
                    boundary = None
                    if 'multipart' in ctype.lower():
                        for t in ctype.split(';'):
                            t = t.strip()
                            if t.lower().startswith('boundary='):
                                boundary = t.split('=', 1)[1].strip().strip('"')
                                break
                    if not boundary:
                        boundary = 'frame'
                    bnd = ('--' + boundary).encode('utf-8')

                    buf = b''
                    for chunk in r.iter_content(chunk_size=2048):
                        if not self.running:
                            break
                        if not chunk:
                            continue
                        buf += chunk

                        while True:
                            start = buf.find(bnd)
                            if start < 0:
                                if len(buf) > 2_000_000:
                                    buf = buf[-1_000_000:]
                                break

                            hdr_end = buf.find(b"\r\n\r\n", start)
                            sep_len = 4
                            if hdr_end < 0:
                                hdr_end = buf.find(b"\n\n", start)
                                if hdr_end >= 0:
                                    sep_len = 2
                            if hdr_end < 0:
                                break

                            headers_block = buf[start+len(bnd):hdr_end]
                            content_length = None
                            for line in headers_block.splitlines():
                                ln = line.strip().lower()
                                if ln.startswith(b'content-length:'):
                                    try:
                                        content_length = int(line.split(b':', 1)[1].strip())
                                    except Exception:
                                        content_length = None
                                    break

                            payload_start = hdr_end + sep_len
                            if content_length is not None:
                                end = payload_start + content_length
                                if len(buf) < end:
                                    break
                                jpeg_bytes = buf[payload_start:end]
                                buf = buf[end:]
                            else:
                                next_b = buf.find(bnd, payload_start)
                                if next_b < 0:
                                    break
                                jpeg_bytes = buf[payload_start:next_b]
                                buf = buf[next_b:]

                            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)

                            flags = cv.IMREAD_COLOR
                            if self.width_scale <= 0.5 and self.height_scale <= 0.5:
                                flags = getattr(cv, "IMREAD_REDUCED_COLOR_2", cv.IMREAD_COLOR)
                            elif self.width_scale <= 0.25 and self.height_scale <= 0.25:
                                flags = getattr(cv, "IMREAD_REDUCED_COLOR_4", cv.IMREAD_COLOR)

                            img = cv.imdecode(arr, flags)
                            if img is None:
                                continue

                            if flags == cv.IMREAD_COLOR and (self.width_scale != 1.0 or self.height_scale != 1.0):
                                h, w = img.shape[:2]
                                img = cv.resize(
                                    img,
                                    (int(w * self.width_scale), int(h * self.height_scale)),
                                    interpolation=cv.INTER_AREA
                                )

                            if not self.lock.acquire(blocking=False):
                                continue
                            try:
                                self.frame = img
                            finally:
                                self.lock.release()

            except Exception as e:
                if self.running:
                    print("⚠ MJPEG reconnect due to:", e, "on", self.url)
                time.sleep(0.25)

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


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
        self.http_cam1 = None
        self.http_cam2 = None
        self.http_timer = None
        
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
        
        overlay_text = f"FPS: {self.fps:.1f} | Frames: {self.frame_count} | {self.connection_status}"
        overlay_text += f"\n{cv_info}"
        
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
    """Video streaming widget with RTSP support - Main camera only in SwitchView"""
    
    def __init__(self, 
                 rtsp_url="rtsp://192.168.1.99:1234", 
                 rtsp_url_cam1="rtsp://localhost:8554/camera1",
                 rtsp_url_cam2="rtsp://localhost:8554/camera2",
                 width_scale=0.5, 
                 height_scale=0.5):
        super().__init__()
        
        # Store URLs
        self.rtsp_url = rtsp_url
        self.rtsp_url_cam1 = rtsp_url_cam1
        self.rtsp_url_cam2 = rtsp_url_cam2
        self.width_scale = width_scale
        self.height_scale = height_scale
        
        # Camera instances
        self.rtsp_camera = None  # Main camera (for display)
        self.rtsp_cam1 = None    # Camera 1 (for AI only)
        self.rtsp_cam2 = None    # Camera 2 (for AI only)
        
        # Start all streams
        self.start_stream(self.rtsp_url, self.rtsp_url_cam1, self.rtsp_url_cam2,
                          self.width_scale, self.height_scale)
        
    def start_stream(self, 
                     rtsp_url="rtsp://192.168.1.99:1234",
                     rtsp_url_cam1="rtsp://localhost:8554/camera1",
                     rtsp_url_cam2="rtsp://localhost:8554/camera2",
                     width_scale=None, height_scale=None):
        """Start all RTSP streams (main + cam1 + cam2)"""
        
        # Update parameters if provided
        if rtsp_url:
            self.rtsp_url = rtsp_url
        if rtsp_url_cam1:
            self.rtsp_url_cam1 = rtsp_url_cam1
        if rtsp_url_cam2:
            self.rtsp_url_cam2 = rtsp_url_cam2
        if width_scale is not None:
            self.width_scale = width_scale
        if height_scale is not None:
            self.height_scale = height_scale
    
        # Validate URLs
        if not self.rtsp_url:
            print("❌ No main RTSP URL provided")
            return False
        if not self.rtsp_url_cam1 or not self.rtsp_url_cam2:
            print("❌ Camera 1 and 2 RTSP URLs are required")
            return False
    
        print(f"🎬 Starting Main RTSP: {self.rtsp_url}")
        print(f"🎬 Starting Camera 1 RTSP: {self.rtsp_url_cam1}")
        print(f"🎬 Starting Camera 2 RTSP: {self.rtsp_url_cam2}")
    
        if cv is None:
            print("❌ OpenCV not available")
            self.set_no_signal_message("OpenCV Not Available")
            return False
    
        try:
            # Stop existing streams if any
            if self.rtsp_camera:
                self.stop_stream()
    
            # ---------- Main RTSP Camera (for SwitchView display) ----------
            self.rtsp_camera = RTSPCamera(
                self.rtsp_url, 
                "",
                "",
                self.width_scale, 
                self.height_scale
            )
            # IMPORTANT: Only RTSP camera updates the main display
            self.rtsp_camera.frame_ready.connect(self.update_frame)
            self.rtsp_camera.connection_status_changed.connect(self.update_connection_status)
            self.rtsp_camera.start()
            print("✅ Main RTSP camera started")
    
            # ---------- Camera 1 RTSP (for AI detection only) ----------
            self.rtsp_cam1 = RTSPCamera(
                self.rtsp_url_cam1,
                "", 
                "",  
                self.width_scale,
                self.height_scale
            )
            # NOT connected to update_frame (runs in background for AI only)
            self.rtsp_cam1.start()
            print("✅ Camera 1 RTSP started (for AI detection)")
            
            # ---------- Camera 2 RTSP (for AI detection only) ----------
            self.rtsp_cam2 = RTSPCamera(
                self.rtsp_url_cam2,
                "", 
                "", 
                self.width_scale,
                self.height_scale
            )
            # NOT connected to update_frame (runs in background for AI only)
            self.rtsp_cam2.start()
            print("✅ Camera 2 RTSP started (for AI detection)")
    
            return True
    
        except Exception as e:
            print(f"❌ Error starting RTSP streams: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_all_frames(self):
        """Get frames from all cameras (Main + Camera1 + Camera2) for AI processing"""
        frames = {}
        
        # Main RTSP camera
        if self.rtsp_camera:
            frame = self.rtsp_camera.get_frame()
            if frame is not None:
                frames['main'] = frame
        
        # RTSP Camera 1 (for AI only)
        if hasattr(self, 'rtsp_cam1') and self.rtsp_cam1:
            frame = self.rtsp_cam1.get_frame()
            if frame is not None:
                frames['camera1'] = frame
        
        # RTSP Camera 2 (for AI only)
        if hasattr(self, 'rtsp_cam2') and self.rtsp_cam2:
            frame = self.rtsp_cam2.get_frame()
            if frame is not None:
                frames['camera2'] = frame
        
        return frames
    
    def stop_stream(self):
        """Stop all RTSP streams"""
        print("🛑 Stopping all streams...")
        
        # Stop main camera
        if self.rtsp_camera:
            self.rtsp_camera.stop()
            self.rtsp_camera = None
            print("✅ Main camera stopped")
        
        # Stop camera 1
        if hasattr(self, 'rtsp_cam1') and self.rtsp_cam1:
            self.rtsp_cam1.stop()
            self.rtsp_cam1 = None
            print("✅ Camera 1 stopped")
        
        # Stop camera 2
        if hasattr(self, 'rtsp_cam2') and self.rtsp_cam2:
            self.rtsp_cam2.stop()
            self.rtsp_cam2 = None
            print("✅ Camera 2 stopped")
        
        # Clear display
        self.clear_video()
    
    @pyqtSlot(str)
    def update_connection_status(self, status):
        """Update connection status"""
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