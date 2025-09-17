"""
Waypoint Management System
Handles waypoint creation, editing, and persistence
"""

import json
import os
from typing import List, Dict, Optional, Tuple
from config.settings import FILE_PATHS


class WaypointManager:
    """Manages waypoint operations with JSON persistence."""
    
    def __init__(self):
        self.waypoints = []
        self.coordinates_file = FILE_PATHS['coordinates_file']
        self._ensure_data_directory()
    
    def _ensure_data_directory(self):
        """Ensure data directory exists."""
        self.coordinates_file.parent.mkdir(parents=True, exist_ok=True)
    
    def add_waypoint(self, x: float, y: float) -> int:
        """Add a new waypoint and return its index."""
        waypoint = {
            'position': (x, y),
            'orientation': 0.0,
            'yaw_enable': False,
            'landing': False,
            'added': False  # Not confirmed yet
        }
        self.waypoints.append(waypoint)
        return len(self.waypoints) - 1
    
    def update_waypoint(self, index: int, **kwargs) -> bool:
        """Update waypoint properties."""
        if not self._is_valid_index(index):
            return False
        
        self.waypoints[index].update(kwargs)
        
        # Auto-save if waypoint is marked as added
        if self.waypoints[index].get('added', False):
            self.save_to_json()
        
        return True
    
    def delete_waypoint(self, index: int) -> bool:
        """Delete a waypoint."""
        if not self._is_valid_index(index):
            return False
        
        del self.waypoints[index]
        self.save_to_json()
        return True
    
    def get_waypoint(self, index: int) -> Optional[Dict]:
        """Get a specific waypoint."""
        if not self._is_valid_index(index):
            return None
        return self.waypoints[index].copy()
    
    def get_all_waypoints(self) -> List[Dict]:
        """Get all waypoints."""
        return [wp.copy() for wp in self.waypoints]
    
    def get_added_waypoints(self) -> List[Dict]:
        """Get only confirmed/added waypoints."""
        return [wp.copy() for wp in self.waypoints if wp.get('added', False)]
    
    def clear_all_waypoints(self):
        """Clear all waypoints."""
        self.waypoints.clear()
        self.save_to_json()
    
    def clear_unadded_waypoints(self):
        """Clear waypoints that haven't been confirmed."""
        self.waypoints = [wp for wp in self.waypoints if wp.get('added', False)]
    
    def save_to_json(self) -> bool:
        """Save added waypoints to JSON file."""
        try:
            # Filter only added waypoints
            added_waypoints = self.get_added_waypoints()
            
            # Convert to the expected format: [x, y, orientation, yaw_enable, landing]
            coordinates_array = []
            for wp in added_waypoints:
                x, y = wp['position']
                coordinates_array.append([
                    float(x),
                    float(y),
                    float(wp['orientation']),
                    int(wp['yaw_enable']),
                    int(wp['landing'])
                ])
            
            # Save to JSON file
            with open(self.coordinates_file, 'w') as f:
                json.dump(coordinates_array, f, indent=2)
            
            print(f"Saved {len(coordinates_array)} waypoints to {self.coordinates_file}")
            return True
            
        except Exception as e:
            print(f"Error saving waypoints to JSON: {e}")
            return False
    
    def load_from_json(self) -> bool:
        """Load waypoints from JSON file with better error handling."""
        try:
            if not self.coordinates_file.exists():
                print(f"Creating default coordinates file at {self.coordinates_file}")
                # Create empty coordinates file
                with open(self.coordinates_file, 'w') as f:
                    json.dump([], f, indent=2)
                return True
            
            with open(self.coordinates_file, 'r') as f:
                content = f.read().strip()
                
            if not content:
                print(f"Coordinates file is empty, initializing with empty array")
                # File exists but empty, create empty array
                coordinates_array = []
                # Write empty array to fix the file
                with open(self.coordinates_file, 'w') as f:
                    json.dump([], f, indent=2)
            else:
                try:
                    coordinates_array = json.loads(content)
                except json.JSONDecodeError as json_error:
                    print(f"Invalid JSON in coordinates file: {json_error}")
                    print("Creating new empty coordinates file")
                    coordinates_array = []
                    # Backup corrupted file and create new one
                    backup_name = str(self.coordinates_file) + '.backup'
                    import shutil
                    shutil.move(str(self.coordinates_file), backup_name)
                    with open(self.coordinates_file, 'w') as f:
                        json.dump([], f, indent=2)
            
            # Clear existing waypoints
            self.waypoints.clear()
            
            # Convert from JSON format to waypoint objects
            for coord in coordinates_array:
                if isinstance(coord, list) and len(coord) >= 5:
                    try:
                        waypoint = {
                            'position': (float(coord[0]), float(coord[1])),
                            'orientation': float(coord[2]),
                            'yaw_enable': bool(coord[3]),
                            'landing': bool(coord[4]),
                            'added': True  # All loaded waypoints are considered added
                        }
                        self.waypoints.append(waypoint)
                    except (ValueError, TypeError) as e:
                        print(f"Skipping invalid waypoint data: {coord} - {e}")
                        continue
            
            print(f"Loaded {len(self.waypoints)} waypoints from {self.coordinates_file}")
            return True
            
        except Exception as e:
            print(f"Error loading waypoints from JSON: {e}")
            # Create empty file on any error
            try:
                with open(self.coordinates_file, 'w') as f:
                    json.dump([], f, indent=2)
                print("Created new empty coordinates file")
            except Exception as create_error:
                print(f"Failed to create new coordinates file: {create_error}")
            
            self.waypoints.clear()
            return False
    
    def export_to_format(self, format_type: str = "json") -> Optional[str]:
        """Export waypoints to different formats."""
        added_waypoints = self.get_added_waypoints()
        
        if format_type.lower() == "json":
            return self._export_to_json(added_waypoints)
        elif format_type.lower() == "csv":
            return self._export_to_csv(added_waypoints)
        elif format_type.lower() == "mavlink":
            return self._export_to_mavlink(added_waypoints)
        else:
            print(f"Unsupported export format: {format_type}")
            return None
    
    def _export_to_json(self, waypoints: List[Dict]) -> str:
        """Export waypoints to JSON string."""
        export_data = {
            "waypoints": [],
            "metadata": {
                "count": len(waypoints),
                "format_version": "1.0"
            }
        }
        
        for i, wp in enumerate(waypoints):
            x, y = wp['position']
            export_data["waypoints"].append({
                "id": i + 1,
                "position": {"x": x, "y": y},
                "orientation": wp['orientation'],
                "yaw_enable": wp['yaw_enable'],
                "landing": wp['landing']
            })
        
        return json.dumps(export_data, indent=2)
    
    def _export_to_csv(self, waypoints: List[Dict]) -> str:
        """Export waypoints to CSV string."""
        csv_lines = ["ID,X,Y,Orientation,YawEnable,Landing"]
        
        for i, wp in enumerate(waypoints):
            x, y = wp['position']
            csv_lines.append(
                f"{i+1},{x:.6f},{y:.6f},{wp['orientation']:.6f},"
                f"{int(wp['yaw_enable'])},{int(wp['landing'])}"
            )
        
        return "\n".join(csv_lines)
    
    def _export_to_mavlink(self, waypoints: List[Dict]) -> str:
        """Export waypoints to MAVLink mission format."""
        mission_items = []
        
        for i, wp in enumerate(waypoints):
            x, y = wp['position']
            
            # MAVLink mission item format
            mission_item = {
                "seq": i,
                "command": 16,  # MAV_CMD_NAV_WAYPOINT
                "param1": 0,    # Hold time
                "param2": 0,    # Acceptance radius
                "param3": 0,    # Pass through
                "param4": wp['orientation'] if wp['yaw_enable'] else 0,  # Yaw
                "x": x,         # Latitude (in local coordinates)
                "y": y,         # Longitude (in local coordinates)
                "z": 0,         # Altitude
                "autocontinue": 1,
                "frame": 1      # Local frame
            }
            
            mission_items.append(mission_item)
            
            # Add landing command if specified
            if wp['landing']:
                landing_item = {
                    "seq": len(mission_items),
                    "command": 21,  # MAV_CMD_NAV_LAND
                    "param1": 0,
                    "param2": 0,
                    "param3": 0,
                    "param4": 0,
                    "x": x,
                    "y": y,
                    "z": 0,
                    "autocontinue": 1,
                    "frame": 1
                }
                mission_items.append(landing_item)
        
        return json.dumps({"mission": mission_items}, indent=2)
    
    def import_from_format(self, data: str, format_type: str = "json") -> bool:
        """Import waypoints from different formats."""
        try:
            if format_type.lower() == "json":
                return self._import_from_json(data)
            elif format_type.lower() == "csv":
                return self._import_from_csv(data)
            else:
                print(f"Unsupported import format: {format_type}")
                return False
        except Exception as e:
            print(f"Error importing waypoints: {e}")
            return False
    
    def _import_from_json(self, json_data: str) -> bool:
        """Import waypoints from JSON string."""
        data = json.loads(json_data)
        
        if "waypoints" in data:
            # New format
            waypoints_data = data["waypoints"]
        else:
            # Old format (array of arrays)
            waypoints_data = data
        
        self.waypoints.clear()
        
        for wp_data in waypoints_data:
            if isinstance(wp_data, list):
                # Old format: [x, y, orientation, yaw_enable, landing]
                if len(wp_data) >= 5:
                    waypoint = {
                        'position': (float(wp_data[0]), float(wp_data[1])),
                        'orientation': float(wp_data[2]),
                        'yaw_enable': bool(wp_data[3]),
                        'landing': bool(wp_data[4]),
                        'added': True
                    }
                    self.waypoints.append(waypoint)
            else:
                # New format: dictionary
                waypoint = {
                    'position': (
                        float(wp_data['position']['x']), 
                        float(wp_data['position']['y'])
                    ),
                    'orientation': float(wp_data['orientation']),
                    'yaw_enable': bool(wp_data['yaw_enable']),
                    'landing': bool(wp_data['landing']),
                    'added': True
                }
                self.waypoints.append(waypoint)
        
        return True
    
    def _import_from_csv(self, csv_data: str) -> bool:
        """Import waypoints from CSV string."""
        lines = csv_data.strip().split('\n')
        
        # Skip header
        if lines[0].startswith('ID,'):
            lines = lines[1:]
        
        self.waypoints.clear()
        
        for line in lines:
            parts = line.split(',')
            if len(parts) >= 6:
                waypoint = {
                    'position': (float(parts[1]), float(parts[2])),
                    'orientation': float(parts[3]),
                    'yaw_enable': bool(int(parts[4])),
                    'landing': bool(int(parts[5])),
                    'added': True
                }
                self.waypoints.append(waypoint)
        
        return True
    
    def get_statistics(self) -> Dict:
        """Get waypoint statistics."""
        total = len(self.waypoints)
        added = len(self.get_added_waypoints())
        pending = total - added
        
        with_yaw = sum(1 for wp in self.waypoints if wp['yaw_enable'])
        with_landing = sum(1 for wp in self.waypoints if wp['landing'])
        
        return {
            'total_waypoints': total,
            'added_waypoints': added,
            'pending_waypoints': pending,
            'waypoints_with_yaw': with_yaw,
            'waypoints_with_landing': with_landing
        }
    
    def _is_valid_index(self, index: int) -> bool:
        """Check if index is valid."""
        return 0 <= index < len(self.waypoints)
    
    def validate_waypoint_data(self, waypoint: Dict) -> Tuple[bool, str]:
        """Validate waypoint data integrity."""
        required_fields = ['position', 'orientation', 'yaw_enable', 'landing', 'added']
        
        for field in required_fields:
            if field not in waypoint:
                return False, f"Missing required field: {field}"
        
        # Validate position
        if not isinstance(waypoint['position'], (list, tuple)) or len(waypoint['position']) != 2:
            return False, "Position must be a tuple/list of two numbers"
        
        try:
            float(waypoint['position'][0])
            float(waypoint['position'][1])
        except (ValueError, TypeError):
            return False, "Position coordinates must be numeric"
        
        # Validate orientation
        try:
            orientation = float(waypoint['orientation'])
            if not (-2 * 3.14159 <= orientation <= 2 * 3.14159):
                return False, "Orientation should be in radians between -2π and 2π"
        except (ValueError, TypeError):
            return False, "Orientation must be numeric"
        
        # Validate boolean fields
        if not isinstance(waypoint['yaw_enable'], bool):
            return False, "yaw_enable must be boolean"
        
        if not isinstance(waypoint['landing'], bool):
            return False, "landing must be boolean"
        
        if not isinstance(waypoint['added'], bool):
            return False, "added must be boolean"
        
        return True, "Valid waypoint data"