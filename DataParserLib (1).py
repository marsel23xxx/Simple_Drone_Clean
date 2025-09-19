"""
DroneParser - Simple UDP Drone Telemetry Parser Library

AVAILABLE METHODS:
    # Setup & Control
    parser = DroneParser(port=8889, max_records=1000)  # Create parser
    parser.start(callback=None)                        # Start capturing data
    parser.stop()                                      # Stop capturing
    parser.clear_data()                                # Clear stored data
    
    # Get Data
    parser.get_latest()                                # Get newest packet
    parser.get_all_data()                              # Get all packets
    parser.get_telemetry_data()                        # Get only flight data packets
    
    # Extract Specific Info (pass record or None for latest)
    parser.get_rpy(record=None)                        # Get roll/pitch/yaw data
    parser.get_position(record=None)                   # Get lat/lon/altitude
    parser.get_battery(record=None)                    # Get battery info
    
    # File Operations
    parser.save_data(filename=None)                    # Save to JSON file
    len(parser)                                        # Number of packets captured

BASIC USAGE:
    # Simple capture
    parser = DroneParser()
    parser.start()
    
    # Get latest data
    latest = parser.get_latest()
    rpy = parser.get_rpy()         # Roll/pitch/yaw in rad & degrees
    pos = parser.get_position()    # XYZ position + altitude
    bat = parser.get_battery()     # Battery % and voltage
    
    # Live monitoring with callback
    def on_data(record):
        rpy = parser.get_rpy(record)
        print(f"Yaw: {rpy['yaw_deg']:.1f}°")
    
    parser.start(callback=on_data)
    
    # Save and stop
    parser.save_data("flight.json")
    parser.stop()

Requirements: pip install scapy
"""

import json
import time
import threading
from scapy.all import sniff, UDP, IP
from datetime import datetime
from typing import Callable, Optional, Dict, Any, List

class DroneParser:
    def __init__(self, port: int = 8889, max_records: int = 1000):
        self.port = port
        self.max_records = max_records
        self.data = []
        self.is_running = False
        self.sniff_thread = None
        self.callback = None
    
    def _extract_json(self, data_bytes: bytes) -> Optional[str]:
        """Extract JSON from UDP packet data"""
        try:
            data_str = data_bytes.decode('utf-8', errors='ignore')
            start = data_str.find('{')
            if start == -1:
                return None
            
            brace_count = 0
            for i, char in enumerate(data_str[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return data_str[start:i+1]
            return None
        except:
            return None
    
    def _handle_packet(self, packet):
        """Internal packet handler"""
        if UDP in packet and packet[UDP].dport == self.port:
            data_bytes = bytes(packet[UDP].payload)
            json_str = self._extract_json(data_bytes)
            
            if json_str:
                try:
                    json_data = json.loads(json_str)
                    
                    # Create record
                    record = {
                        'timestamp': datetime.now(),
                        'source_ip': packet[IP].src,
                        'source_port': packet[UDP].sport,
                        'data': json_data
                    }
                    
                    # Add to data list
                    self.data.append(record)
                    if len(self.data) > self.max_records:
                        self.data.pop(0)
                    
                    # Call callback if set
                    if self.callback:
                        self.callback(record)
                        
                except json.JSONDecodeError:
                    pass
    
    def start(self, callback: Optional[Callable] = None) -> None:
        """Start capturing data"""
        if self.is_running:
            return
        
        self.callback = callback
        self.is_running = True
        
        def sniff_worker():
            sniff(filter=f"udp port {self.port}", prn=self._handle_packet, stop_filter=lambda x: not self.is_running)
        
        self.sniff_thread = threading.Thread(target=sniff_worker, daemon=True)
        self.sniff_thread.start()
    
    def stop(self) -> None:
        """Stop capturing data"""
        self.is_running = False
        if self.sniff_thread:
            self.sniff_thread.join(timeout=2)
    
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get the latest record"""
        return self.data[-1] if self.data else None
    
    def get_all_data(self) -> List[Dict[str, Any]]:
        """Get all captured data"""
        return self.data.copy()
    
    def get_telemetry_data(self) -> List[Dict[str, Any]]:
        """Get only telemetry records"""
        return [r for r in self.data if 'altitude' in r['data']]
    
    def get_rpy(self, record: Optional[Dict] = None) -> Optional[Dict[str, float]]:
        """Extract RPY data from a record"""
        if record is None:
            record = self.get_latest()
        
        if not record or 'attitude' not in record['data']:
            return None
        
        attitude = record['data']['attitude']
        if len(attitude) < 3:
            return None
        
        return {
            'roll': attitude[0],
            'pitch': attitude[1], 
            'yaw': attitude[2],
            'roll_deg': attitude[0] * 57.2958,
            'pitch_deg': attitude[1] * 57.2958,
            'yaw_deg': attitude[2] * 57.2958
        }
    
    def get_velocity(self, record: Optional[Dict] = None) -> Optional[Dict[str, float]]:
        """Extract RPY data from a record"""
        if record is None:
            record = self.get_latest()
        
        if not record or 'velocity' not in record['data']:
            return None
        
        velocity = record['data']['velocity']
        if len(velocity) < 3:
            return None
        
        return {
            'xspeed': velocity[0],
            'yspeed': velocity[1], 
            'zspeed': velocity[2],
        }
    
    def get_position(self, record: Optional[Dict] = None) -> Optional[Dict[str, float]]:
        """Extract position data from a record"""
        if record is None:
            record = self.get_latest()
        
        if not record or 'position' not in record['data']:
            return None
        
        position = record['data']['position']
        if len(position) < 3:
            return None
        return {
            'x': position[0],
            'y': position[1], 
            'z': position[2],
        }
    
    def get_battery(self, record: Optional[Dict] = None) -> Optional[Dict[str, float]]:
        """Extract battery data from a record"""
        if record is None:
            record = self.get_latest()
        
        if not record:
            return None
        
        data = record['data']
        return {
            'percentage': data.get('battery_percetage', 0) * 100,
            'voltage': data.get('battery_state')
        }
    
    def save_data(self, filename: Optional[str] = None) -> str:
        """Save all data to JSON file"""
        if filename is None:
            filename = f"drone_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert datetime objects to strings for JSON serialization
        data_copy = []
        for record in self.data:
            record_copy = record.copy()
            record_copy['timestamp'] = record['timestamp'].isoformat()
            data_copy.append(record_copy)
        
        with open(filename, 'w') as f:
            json.dump(data_copy, f, indent=2)
        
        return filename
    
    def clear_data(self) -> None:
        """Clear all stored data"""
        self.data.clear()
    
    def __len__(self) -> int:
        """Return number of records"""
        return len(self.data)

# Example usage functions
def print_data(record):
    """Example callback function"""
    data = record['data']
    timestamp = record['timestamp'].strftime("%H:%M:%S.%f")[:-3]
    
    print(f"\n[{timestamp}] From {record['source_ip']}:{record['source_port']}")
    
    # Orientation RPY data
    if 'attitude' in data and len(data['attitude']) >= 3:
        roll, pitch, yaw = data['attitude'][:3]
        print(f"🧭 Orientation RPY: R={roll:.3f}, P={pitch:.3f}, Y={yaw:.3f} (rad)")
        print(f"    Orientation RPY: R={roll*57.3:.1f}°, P={pitch*57.3:.1f}°, Y={yaw*57.3:.1f}°")
    
    # Position data
    if 'position' in data and len(data['position']) >= 3:
        x, y, z = data['position'][:3]
        print(f"📍 Position XYZ: X={x:.2f}m, Y={y:.2f}m, Z={z:.2f}m")
        
    if 'armed' in data:
        print(f"⚡ Armed: {data['armed']}")

# Simple usage example
if __name__ == "__main__":
    parser = DroneParser(port=8889)
    parser.start()
    
    while True:
        # Get latest data
        latest = parser.get_latest()
        rpy = parser.get_rpy()         # Roll/pitch/yaw in rad & degrees
        pos = parser.get_position()    # XYZ position + altitude
        bat = parser.get_battery()     # Battery % and voltage
        vel = parser.get_velocity()   # Velocity in m/s
        
        print(rpy)
        print(pos)
        print(bat)
        print(vel)
        time.sleep(1)
        
        # Save and stop
        parser.save_data("flight.json")
        parser.stop()