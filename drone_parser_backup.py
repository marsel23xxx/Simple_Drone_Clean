"""
Drone Data Parser
Handles incoming drone telemetry data and provides structured access
"""

import socket
import struct
import threading
import time
from typing import Dict, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass
from config.settings import NETWORK_CONFIG


@dataclass
class DronePosition:
    """Drone position data structure."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    timestamp: float = 0.0


@dataclass
class DroneOrientation:
    """Drone orientation data structure."""
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    timestamp: float = 0.0


@dataclass
class DroneVelocity:
    """Drone velocity data structure."""
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    timestamp: float = 0.0


@dataclass
class DroneBattery:
    """Drone battery data structure."""
    voltage: float = 0.0
    percentage: int = 0
    current: float = 0.0
    timestamp: float = 0.0


@dataclass
class DroneStatus:
    """Drone status data structure."""
    armed: bool = False
    mode: str = "UNKNOWN"
    connected: bool = False
    flight_time: int = 0  # seconds
    timestamp: float = 0.0


class DroneDataParser:
    """Parser for drone telemetry data with multiple protocol support."""
    
    def __init__(self, port: int = 8889, buffer_size: int = 100):
        self.port = port
        self.buffer_size = buffer_size
        self.running = False
        self.connected = False
        
        # Data storage with circular buffers
        self.position_buffer = deque(maxlen=buffer_size)
        self.orientation_buffer = deque(maxlen=buffer_size)
        self.velocity_buffer = deque(buffer_size)
        self.battery_buffer = deque(maxlen=buffer_size)
        self.status_buffer = deque(maxlen=buffer_size)
        
        # Latest data cache
        self.latest_position = DronePosition()
        self.latest_orientation = DroneOrientation()
        self.latest_velocity = DroneVelocity()
        self.latest_battery = DroneBattery()
        self.latest_status = DroneStatus()
        
        # Network components
        self.socket = None
        self.thread = None
        self.lock = threading.Lock()
        
        # Statistics
        self.packets_received = 0
        self.packets_parsed = 0
        self.parse_errors = 0
        self.last_update_time = 0.0
    
    def start(self):
        """Start the drone data parser."""
        if self.running:
            print("Drone parser already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_parser, daemon=True)
        self.thread.start()
        print(f"Drone parser started on port {self.port}")
    
    def stop(self):
        """Stop the drone data parser."""
        self.running = False
        if self.socket:
            self.socket.close()
        if self.thread:
            self.thread.join(timeout=2.0)
        print("Drone parser stopped")
    
    def _run_parser(self):
        """Main parser loop."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('', self.port))
            self.socket.settimeout(1.0)
            
            print(f"Listening for drone data on port {self.port}")
            
            while self.running:
                try:
                    data, addr = self.socket.recvfrom(1024)
                    self.packets_received += 1
                    
                    if not self.connected:
                        self.connected = True
                        print(f"Drone connected from {addr}")
                    
                    self._parse_data(data)
                    self.last_update_time = time.time()
                    
                except socket.timeout:
                    # Check if we lost connection
                    if self.connected and time.time() - self.last_update_time > 5.0:
                        self.connected = False
                        print("Drone connection lost")
                    continue
                except Exception as e:
                    print(f"Error receiving drone data: {e}")
                    self.parse_errors += 1
                    
        except Exception as e:
            print(f"Error starting drone parser: {e}")
        finally:
            if self.socket:
                self.socket.close()
            self.connected = False
    
    def _parse_data(self, data: bytes):
        """Parse incoming drone data."""
        try:
            if len(data) < 4:
                return
            
            # Parse message type (first 4 bytes)
            msg_type = struct.unpack('<I', data[:4])[0]
            payload = data[4:]
            
            if msg_type == 1:  # Position data
                self._parse_position(payload)
            elif msg_type == 2:  # Orientation data
                self._parse_orientation(payload)
            elif msg_type == 3:  # Velocity data
                self._parse_velocity(payload)
            elif msg_type == 4:  # Battery data
                self._parse_battery(payload)
            elif msg_type == 5:  # Status data
                self._parse_status(payload)
            elif msg_type == 0:  # Combined telemetry
                self._parse_combined_telemetry(payload)
            else:
                print(f"Unknown message type: {msg_type}")
                return
            
            self.packets_parsed += 1
            
        except Exception as e:
            print(f"Error parsing drone data: {e}")
            self.parse_errors += 1
    
    def _parse_position(self, payload: bytes):
        """Parse position data."""
        if len(payload) < 12:  # 3 floats
            return
        
        x, y, z = struct.unpack('<fff', payload[:12])
        
        with self.lock:
            position = DronePosition(x, y, z, time.time())
            self.latest_position = position
            self.position_buffer.append(position)
    
    def _parse_orientation(self, payload: bytes):
        """Parse orientation data (roll, pitch, yaw)."""
        if len(payload) < 12:  # 3 floats
            return
        
        roll, pitch, yaw = struct.unpack('<fff', payload[:12])
        
        with self.lock:
            orientation = DroneOrientation(roll, pitch, yaw, time.time())
            self.latest_orientation = orientation
            self.orientation_buffer.append(orientation)
    
    def _parse_velocity(self, payload: bytes):
        """Parse velocity data."""
        if len(payload) < 12:  # 3 floats
            return
        
        vx, vy, vz = struct.unpack('<fff', payload[:12])
        
        with self.lock:
            velocity = DroneVelocity(vx, vy, vz, time.time())
            self.latest_velocity = velocity
            self.velocity_buffer.append(velocity)
    
    def _parse_battery(self, payload: bytes):
        """Parse battery data."""
        if len(payload) < 12:  # float + int + float
            return
        
        voltage, percentage, current = struct.unpack('<fIf', payload[:12])
        
        with self.lock:
            battery = DroneBattery(voltage, percentage, current, time.time())
            self.latest_battery = battery
            self.battery_buffer.append(battery)
    
    def _parse_status(self, payload: bytes):
        """Parse status data."""
        if len(payload) < 20:
            return
        
        armed, mode_len = struct.unpack('<BI', payload[:5])
        
        if len(payload) < 5 + mode_len + 4:
            return
        
        mode = payload[5:5+mode_len].decode('utf-8', errors='ignore')
        flight_time = struct.unpack('<I', payload[5+mode_len:9+mode_len])[0]
        
        with self.lock:
            status = DroneStatus(
                armed=bool(armed),
                mode=mode,
                connected=self.connected,
                flight_time=flight_time,
                timestamp=time.time()
            )
            self.latest_status = status
            self.status_buffer.append(status)
    
    def _parse_combined_telemetry(self, payload: bytes):
        """Parse combined telemetry data."""
        if len(payload) < 48:  # Minimum for combined data
            return
        
        try:
            # Position (12 bytes)
            x, y, z = struct.unpack('<fff', payload[:12])
            
            # Orientation (12 bytes)
            roll, pitch, yaw = struct.unpack('<fff', payload[12:24])
            
            # Velocity (12 bytes)
            vx, vy, vz = struct.unpack('<fff', payload[24:36])
            
            # Battery (12 bytes)
            voltage, percentage, current = struct.unpack('<fIf', payload[36:48])
            
            timestamp = time.time()
            
            with self.lock:
                # Update all data structures
                self.latest_position = DronePosition(x, y, z, timestamp)
                self.latest_orientation = DroneOrientation(roll, pitch, yaw, timestamp)
                self.latest_velocity = DroneVelocity(vx, vy, vz, timestamp)
                self.latest_battery = DroneBattery(voltage, percentage, current, timestamp)
                
                # Add to buffers
                self.position_buffer.append(self.latest_position)
                self.orientation_buffer.append(self.latest_orientation)
                self.velocity_buffer.append(self.latest_velocity)
                self.battery_buffer.append(self.latest_battery)
                
        except Exception as e:
            print(f"Error parsing combined telemetry: {e}")
    
    def get_latest_position(self) -> Optional[DronePosition]:
        """Get the latest position data."""
        with self.lock:
            if self.position_buffer:
                return self.latest_position
        return None
    
    def get_latest_orientation(self) -> Optional[DroneOrientation]:
        """Get the latest orientation data."""
        with self.lock:
            if self.orientation_buffer:
                return self.latest_orientation
        return None
    
    def get_latest_velocity(self) -> Optional[DroneVelocity]:
        """Get the latest velocity data."""
        with self.lock:
            if self.velocity_buffer:
                return self.latest_velocity
        return None
    
    def get_latest_battery(self) -> Optional[DroneBattery]:
        """Get the latest battery data."""
        with self.lock:
            if self.battery_buffer:
                return self.latest_battery
        return None
    
    def get_latest_status(self) -> Optional[DroneStatus]:
        """Get the latest status data."""
        with self.lock:
            if self.status_buffer:
                return self.latest_status
        return None
    
    def get_position_history(self, count: int = 10) -> list:
        """Get recent position history."""
        with self.lock:
            return list(self.position_buffer)[-count:]
    
    def get_orientation_history(self, count: int = 10) -> list:
        """Get recent orientation history."""
        with self.lock:
            return list(self.orientation_buffer)[-count:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get parser statistics."""
        return {
            'connected': self.connected,
            'packets_received': self.packets_received,
            'packets_parsed': self.packets_parsed,
            'parse_errors': self.parse_errors,
            'success_rate': (self.packets_parsed / max(1, self.packets_received)) * 100,
            'last_update': self.last_update_time,
            'position_buffer_size': len(self.position_buffer),
            'orientation_buffer_size': len(self.orientation_buffer),
            'velocity_buffer_size': len(self.velocity_buffer),
            'battery_buffer_size': len(self.battery_buffer),
            'status_buffer_size': len(self.status_buffer)
        }
    
    def is_connected(self) -> bool:
        """Check if drone is connected."""
        return self.connected and (time.time() - self.last_update_time) < 5.0


# Legacy compatibility class (for backward compatibility with existing code)
class DroneParser:
    """Legacy drone parser for backward compatibility."""
    
    def __init__(self, port: int = 8889):
        self.parser = DroneDataParser(port)
        self._latest_data = {}
    
    def start(self):
        """Start the parser."""
        self.parser.start()
    
    def stop(self):
        """Stop the parser."""
        self.parser.stop()
    
    def get_latest(self) -> Optional[Dict]:
        """Get latest data in legacy format."""
        if not self.parser.is_connected():
            return None
        
        # Combine all latest data into legacy format
        position = self.parser.get_latest_position()
        orientation = self.parser.get_latest_orientation()
        velocity = self.parser.get_latest_velocity()
        battery = self.parser.get_latest_battery()
        status = self.parser.get_latest_status()
        
        if not all([position, orientation]):
            return None
        
        return {
            'position': {
                'x': position.x,
                'y': position.y,
                'z': position.z
            },
            'orientation': {
                'roll': orientation.roll,
                'pitch': orientation.pitch,
                'yaw': orientation.yaw
            },
            'velocity': {
                'vx': velocity.vx if velocity else 0.0,
                'vy': velocity.vy if velocity else 0.0,
                'vz': velocity.vz if velocity else 0.0
            },
            'battery': {
                'voltage': battery.voltage if battery else 0.0,
                'percentage': battery.percentage if battery else 0,
                'current': battery.current if battery else 0.0
            },
            'status': {
                'armed': status.armed if status else False,
                'mode': status.mode if status else "UNKNOWN",
                'flight_time': status.flight_time if status else 0
            },
            'timestamp': time.time()
        }
    
    def get_position(self, data: Optional[Dict]) -> Optional[Dict]:
        """Get position from data in legacy format."""
        if data and 'position' in data:
            return data['position']
        return None
    
    def get_rpy(self, data: Optional[Dict]) -> Optional[Dict]:
        """Get roll, pitch, yaw from data in legacy format."""
        if data and 'orientation' in data:
            return data['orientation']
        return None
    
    def get_velocity(self, data: Optional[Dict]) -> Optional[Dict]:
        """Get velocity from data in legacy format."""
        if data and 'velocity' in data:
            return data['velocity']
        return None
    
    def get_battery(self, data: Optional[Dict]) -> Optional[Dict]:
        """Get battery from data in legacy format."""
        if data and 'battery' in data:
            return data['battery']
        return None
    
    def get_status(self, data: Optional[Dict]) -> Optional[Dict]:
        """Get status from data in legacy format."""
        if data and 'status' in data:
            return data['status']
        return None