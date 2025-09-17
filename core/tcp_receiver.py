"""
TCP Data Receiver for Point Cloud Data
Handles incoming point cloud data via TCP connection
"""

import struct
import socket
import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal

from config.settings import NETWORK_CONFIG, POINTCLOUD_CONFIG


class TCPDataReceiver(QObject):
    """TCP server for receiving point cloud data."""
    
    frame_received = pyqtSignal(np.ndarray)
    connection_status = pyqtSignal(bool, str)  # connected, message
    
    def __init__(self):
        super().__init__()
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.connected = False
        
        # Frame counters for performance tracking
        self.received_frame_count = 0
        self.processed_frame_count = 0
        self.displayed_frame_count = 0
        
        # Frame skipping configuration from settings
        self.process_frame_skip = POINTCLOUD_CONFIG['process_frame_skip']
        self.display_frame_skip = POINTCLOUD_CONFIG['display_frame_skip']
    
    def start_server(self):
        """Start TCP server to listen for point cloud data."""
        self.running = True
        tcp_port = NETWORK_CONFIG['tcp_listen_port']
        
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('', tcp_port))
            self.server_socket.listen(1)
            self.server_socket.settimeout(1.0)  # 1 second timeout for accept
            
            self.connection_status.emit(False, f"TCP server listening on port {tcp_port}")
            print(f"TCP server listening on port {tcp_port}")
            
            while self.running:
                try:
                    print("Waiting for point cloud data connection...")
                    self.client_socket, client_address = self.server_socket.accept()
                    self.client_socket.settimeout(2.0)  # 2 second timeout for recv
                    
                    self.connected = True
                    self.connection_status.emit(True, f"Point cloud source connected from {client_address[0]}")
                    print(f"Point cloud source connected from {client_address}")
                    
                    # Handle client connection
                    self.handle_client_connection()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"TCP accept error: {e}")
                        self.connection_status.emit(False, f"Accept error: {e}")
                
                if self.client_socket:
                    self.client_socket.close()
                    self.client_socket = None
                    self.connected = False
                    self.connection_status.emit(False, "Point cloud source disconnected")
                
        except Exception as e:
            print(f"TCP server error: {e}")
            self.connection_status.emit(False, f"Server error: {e}")
        finally:
            self.cleanup_server()
    
    def handle_client_connection(self):
        """Handle data from connected client."""
        buffer = b''
        
        while self.running and self.connected:
            try:
                data = self.client_socket.recv(8192)
                if not data:
                    print("Client disconnected")
                    break
                
                buffer += data
                
                # Process complete messages in buffer
                while len(buffer) >= 4:
                    # Read points count
                    points_count = struct.unpack('<I', buffer[:4])[0]
                    expected_size = 4 + points_count * 3 * 4  # header + points data
                    
                    if len(buffer) >= expected_size:
                        # Extract complete message
                        message = buffer[:expected_size]
                        buffer = buffer[expected_size:]
                        
                        # Process the point cloud data
                        self.process_binary_data(message)
                    else:
                        # Wait for more data
                        break
                        
            except socket.timeout:
                continue
            except Exception as e:
                print(f"TCP receive error: {e}")
                break
    
    def process_binary_data(self, data):
        """Process binary point cloud data with frame skipping."""
        try:
            # Assume data format: [points_count (4 bytes)] + [points_data]
            if len(data) < 4:
                return
            
            points_count = struct.unpack('<I', data[:4])[0]
            points_data = data[4:]
            
            expected_size = points_count * 3 * 4  # 3 floats per point
            if len(points_data) != expected_size:
                print(f"Data size mismatch: expected {expected_size}, got {len(points_data)}")
                return
            
            points = np.frombuffer(points_data, dtype=np.float32).reshape(-1, 3)
            
            self.received_frame_count += 1
            
            # Process frame skip logic
            should_process = (self.received_frame_count % self.process_frame_skip) == 0
            
            if should_process:
                self.processed_frame_count += 1
                
                # Display frame skip logic
                should_display = (self.processed_frame_count % self.display_frame_skip) == 0
                
                if should_display:
                    self.displayed_frame_count += 1
                    self.frame_received.emit(points)
                    
        except Exception as e:
            print(f"Error processing binary data: {e}")
    
    def cleanup_server(self):
        """Clean up server resources."""
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
        
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
        
        self.connected = False
        self.connection_status.emit(False, "TCP server stopped")
    
    def stop_server(self):
        """Stop the TCP server."""
        self.running = False
    
    def get_stats(self):
        """Get frame processing statistics."""
        return {
            'received': self.received_frame_count,
            'processed': self.processed_frame_count,
            'displayed': self.displayed_frame_count
        }


class TCPServerThread(QThread):
    """Thread to run TCP server."""
    
    def __init__(self, tcp_receiver):
        super().__init__()
        self.tcp_receiver = tcp_receiver
    
    def run(self):
        """Run the TCP server."""
        self.tcp_receiver.start_server()
    
    def stop(self):
        """Stop the TCP server thread."""
        self.tcp_receiver.stop_server()
        self.quit()
        self.wait()