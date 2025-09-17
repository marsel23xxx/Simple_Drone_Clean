"""
Drone Status Widget
Widget untuk menampilkan status lengkap drone dengan visualisasi real-time
"""

import math
import time
from PyQt5.QtWidgets import QWidget, QLabel, QProgressBar, QVBoxLayout, QHBoxLayout, QGridLayout
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPixmap, QPainter, QTransform, QColor, QFont, QPen, QBrush

from config.settings import ASSET_PATHS


class DroneStatusWidget(QWidget):
    """Widget untuk menampilkan status lengkap drone dengan visualisasi real-time."""
    
    # Signals untuk notifikasi status changes
    battery_low = pyqtSignal(int)  # Emit ketika battery rendah
    connection_lost = pyqtSignal()
    emergency_mode = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Drone status data
        self.drone_data = {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'lat': 0.0, 'lon': 0.0},
            'velocity': {'vx': 0.0, 'vy': 0.0, 'vz': 0.0, 'speed': 0.0},
            'attitude': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
            'battery': {'percentage': 0, 'voltage': 0.0, 'current': 0.0},
            'flight_time': 0,
            'mode': 'Unknown',
            'armed': False,
            'connected': False,
            'gps_fix': False,
            'satellites': 0
        }
        
        # Widget references (akan di-set dari main window)
        self.drone_top_view = None
        self.drone_side_view = None 
        self.drone_bottom_view = None
        self.battery_widget = None
        self.position_labels = {}
        self.speed_labels = {}
        self.status_labels = {}
        
        # Original images untuk rotation
        self.original_top_image = None
        self.original_side_image = None
        self.original_bottom_image = None
        
        # Animation untuk smooth updates
        self.animations = {}
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_animations)
        self.update_timer.start(50)  # 20 FPS untuk smooth animation
        
        # Load drone images
        self.load_drone_images()
        
        # Status tracking untuk alerts
        self.last_battery_alert = 0
        self.connection_timeout = 0
        
    def set_ui_widgets(self, ui_refs):
        """Set references ke UI widgets dari main window.
        
        Args:
            ui_refs: Dictionary dengan widget references
        """
        self.drone_top_view = ui_refs.get('DroneTopView')
        self.drone_side_view = ui_refs.get('DroneSideView') 
        self.drone_bottom_view = ui_refs.get('DroneBottomView')
        self.battery_widget = ui_refs.get('DroneBattery')
        
        # Position labels
        self.position_labels = {
            'x': ui_refs.get('DronePositionX'),
            'y': ui_refs.get('DronePositionY'),
            'height': ui_refs.get('DroneHeight')
        }
        
        # Speed labels
        self.speed_labels = {
            'vx': ui_refs.get('DroneSpeedX'),
            'vy': ui_refs.get('DroneSpeedY'), 
            'vz': ui_refs.get('DroneSpeedZ')
        }
        
        # Status labels
        self.status_labels = {
            'mode': ui_refs.get('DroneMode'),
            'flight_time': ui_refs.get('DroneFlightTime')
        }
        
        # Initialize displays
        self.update_all_displays()
    
    def load_drone_images(self):
        """Load original drone images untuk rotation animations."""
        try:
            if ASSET_PATHS.get('drone_top') and ASSET_PATHS['drone_top'].exists():
                self.original_top_image = QPixmap(str(ASSET_PATHS['drone_top']))
                
            if ASSET_PATHS.get('drone_side') and ASSET_PATHS['drone_side'].exists():
                self.original_side_image = QPixmap(str(ASSET_PATHS['drone_side']))
                
            if ASSET_PATHS.get('drone_bottom') and ASSET_PATHS['drone_bottom'].exists():
                self.original_bottom_image = QPixmap(str(ASSET_PATHS['drone_bottom']))
                
        except Exception as e:
            print(f"Error loading drone images: {e}")
    
    def update_drone_data(self, data_dict):
        """Update drone data dan refresh displays.
        
        Args:
            data_dict: Dictionary dengan drone data terbaru
        """
        # Update data
        for category, values in data_dict.items():
            if category in self.drone_data:
                self.drone_data[category].update(values)
        
        # Set connection status
        self.drone_data['connected'] = True
        self.connection_timeout = time.time()
        
        # Update displays
        self.update_all_displays()
        
        # Check for alerts
        self.check_alerts()
    
    def update_position(self, x=None, y=None, z=None, lat=None, lon=None):
        """Update posisi drone."""
        if x is not None:
            self.drone_data['position']['x'] = x
        if y is not None:
            self.drone_data['position']['y'] = y
        if z is not None:
            self.drone_data['position']['z'] = z
        if lat is not None:
            self.drone_data['position']['lat'] = lat
        if lon is not None:
            self.drone_data['position']['lon'] = lon
            
        self.update_position_display()
    
    def update_velocity(self, vx=None, vy=None, vz=None):
        """Update kecepatan drone."""
        if vx is not None:
            self.drone_data['velocity']['vx'] = vx
        if vy is not None:
            self.drone_data['velocity']['vy'] = vy
        if vz is not None:
            self.drone_data['velocity']['vz'] = vz
            
        # Calculate total speed
        vx = self.drone_data['velocity']['vx']
        vy = self.drone_data['velocity']['vy']
        vz = self.drone_data['velocity']['vz']
        self.drone_data['velocity']['speed'] = math.sqrt(vx**2 + vy**2 + vz**2)
        
        self.update_speed_display()
    
    def update_attitude(self, roll=None, pitch=None, yaw=None):
        """Update attitude drone (roll, pitch, yaw)."""
        if roll is not None:
            self.drone_data['attitude']['roll'] = roll
        if pitch is not None:
            self.drone_data['attitude']['pitch'] = pitch
        if yaw is not None:
            self.drone_data['attitude']['yaw'] = yaw
            
        self.update_drone_orientation()
    
    def update_battery(self, percentage=None, voltage=None, current=None):
        """Update battery status."""
        if percentage is not None:
            self.drone_data['battery']['percentage'] = percentage
        if voltage is not None:
            self.drone_data['battery']['voltage'] = voltage
        if current is not None:
            self.drone_data['battery']['current'] = current
            
        self.update_battery_display()
    
    def update_flight_status(self, mode=None, armed=None, flight_time=None, gps_fix=None, satellites=None):
        """Update flight status."""
        if mode is not None:
            self.drone_data['mode'] = mode
        if armed is not None:
            self.drone_data['armed'] = armed
        if flight_time is not None:
            self.drone_data['flight_time'] = flight_time
        if gps_fix is not None:
            self.drone_data['gps_fix'] = gps_fix
        if satellites is not None:
            self.drone_data['satellites'] = satellites
            
        self.update_status_display()
    
    def update_all_displays(self):
        """Update semua displays."""
        self.update_position_display()
        self.update_speed_display()
        self.update_battery_display()
        self.update_status_display()
        self.update_drone_orientation()
    
    def update_position_display(self):
        """Update position display labels."""
        pos = self.drone_data['position']
        
        if self.position_labels.get('x'):
            self.position_labels['x'].setText(f"[{pos['x']:.2f}] m")
            
        if self.position_labels.get('y'):
            self.position_labels['y'].setText(f"[{pos['y']:.2f}] m")
            
        if self.position_labels.get('height'):
            self.position_labels['height'].setText(f"{pos['z']:.2f} meter")
    
    def update_speed_display(self):
        """Update speed display labels."""
        vel = self.drone_data['velocity']
        
        if self.speed_labels.get('vx'):
            self.speed_labels['vx'].setText(f"[{vel['vx']:.2f}] m/s")
            
        if self.speed_labels.get('vy'):
            self.speed_labels['vy'].setText(f"[{vel['vy']:.2f}] m/s")
            
        if self.speed_labels.get('vz'):
            self.speed_labels['vz'].setText(f"[{vel['vz']:.2f}] m/s")
    
    def update_battery_display(self):
        """Update battery display dengan color coding."""
        battery = self.drone_data['battery']
        
        if self.battery_widget:
            self.battery_widget.setValue(battery['percentage'])
            
            # Color coding berdasarkan battery level
            if battery['percentage'] < 20:
                color = "#e63946"  # Red
            elif battery['percentage'] < 50:
                color = "#f77f00"  # Orange
            elif battery['percentage'] < 80:
                color = "#ffb700"  # Yellow
            else:
                color = "#3ddb55"  # Green
                
            self.battery_widget.setStyleSheet(f"""
                QProgressBar::chunk {{ 
                    background-color: {color};
                    border-radius: 5px;
                }}
                QProgressBar {{
                    border: 1px solid #555;
                    border-radius: 5px;
                    text-align: center;
                    background-color: #333;
                }}
            """)
    
    def update_status_display(self):
        """Update status displays."""
        if self.status_labels.get('mode'):
            mode_text = self.drone_data['mode']
            if self.drone_data['armed']:
                mode_text += " (ARMED)"
            self.status_labels['mode'].setText(mode_text)
            
            # Color coding berdasarkan mode
            if self.drone_data['armed']:
                self.status_labels['mode'].setStyleSheet("color: #e63946;")  # Red untuk armed
            else:
                self.status_labels['mode'].setStyleSheet("color: #3ddb55;")  # Green untuk disarmed
        
        if self.status_labels.get('flight_time'):
            ft = self.drone_data['flight_time']
            minutes = ft // 60
            seconds = ft % 60
            self.status_labels['flight_time'].setText(f"{minutes:02d}:{seconds:02d}")
    
    def update_drone_orientation(self):
        """Update drone orientation views berdasarkan attitude."""
        attitude = self.drone_data['attitude']
        
        # Update top view dengan yaw rotation
        if self.drone_top_view and self.original_top_image:
            rotated_image = self.rotate_image(self.original_top_image, -math.degrees(attitude['yaw']))
            self.drone_top_view.setPixmap(rotated_image)
        
        # Update side view dengan pitch rotation
        if self.drone_side_view and self.original_side_image:
            rotated_image = self.rotate_image(self.original_side_image, math.degrees(attitude['pitch']))
            self.drone_side_view.setPixmap(rotated_image)
        
        # Update bottom view dengan roll rotation
        if self.drone_bottom_view and self.original_bottom_image:
            rotated_image = self.rotate_image(self.original_bottom_image, math.degrees(attitude['roll']))
            self.drone_bottom_view.setPixmap(rotated_image)
    
    def rotate_image(self, original_pixmap, angle_degrees):
        """Rotate image dengan smooth transformation."""
        if original_pixmap.isNull():
            return original_pixmap
            
        transform = QTransform()
        transform.rotate(angle_degrees)
        
        rotated = original_pixmap.transformed(transform, Qt.SmoothTransformation)
        return rotated
    
    def check_alerts(self):
        """Check untuk alerts dan emit signals jika perlu."""
        battery_pct = self.drone_data['battery']['percentage']
        
        # Battery low alert
        if battery_pct < 20 and time.time() - self.last_battery_alert > 30:  # Alert setiap 30 detik
            self.battery_low.emit(battery_pct)
            self.last_battery_alert = time.time()
        
        # Connection timeout check
        if time.time() - self.connection_timeout > 5:  # 5 detik timeout
            if self.drone_data['connected']:
                self.drone_data['connected'] = False
                self.connection_lost.emit()
        
        # Emergency mode check
        if (self.drone_data['mode'].lower() in ['emergency', 'failsafe', 'rtl'] and 
            self.drone_data['armed']):
            self.emergency_mode.emit()
    
    def update_animations(self):
        """Update smooth animations untuk status changes."""
        # Smooth battery bar animation sudah handled oleh Qt
        # Bisa ditambahkan custom animations lain disini
        pass
    
    def get_drone_status_summary(self):
        """Get summary status drone untuk debugging."""
        return {
            'position': f"({self.drone_data['position']['x']:.1f}, {self.drone_data['position']['y']:.1f}, {self.drone_data['position']['z']:.1f})",
            'speed': f"{self.drone_data['velocity']['speed']:.1f} m/s",
            'battery': f"{self.drone_data['battery']['percentage']}%",
            'mode': self.drone_data['mode'],
            'armed': self.drone_data['armed'],
            'connected': self.drone_data['connected'],
            'gps': f"{self.drone_data['satellites']} sats" if self.drone_data['gps_fix'] else "No GPS"
        }
    
    def reset_drone_data(self):
        """Reset semua drone data ke default values."""
        self.drone_data = {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'lat': 0.0, 'lon': 0.0},
            'velocity': {'vx': 0.0, 'vy': 0.0, 'vz': 0.0, 'speed': 0.0},
            'attitude': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
            'battery': {'percentage': 0, 'voltage': 0.0, 'current': 0.0},
            'flight_time': 0,
            'mode': 'Disconnected',
            'armed': False,
            'connected': False,
            'gps_fix': False,
            'satellites': 0
        }
        self.update_all_displays()
    
    def set_connection_status(self, connected):
        """Set connection status manually."""
        self.drone_data['connected'] = connected
        if connected:
            self.connection_timeout = time.time()
        else:
            self.drone_data['mode'] = 'Disconnected'
        self.update_status_display()


class DroneAttitudeIndicator(QWidget):
    """Custom widget untuk attitude indicator yang lebih advanced."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 100)
        
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
    def update_attitude(self, roll, pitch, yaw):
        """Update attitude dan repaint."""
        self.roll = roll
        self.pitch = pitch  
        self.yaw = yaw
        self.update()
    
    def paintEvent(self, event):
        """Paint attitude indicator."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        center_x = self.width() / 2
        center_y = self.height() / 2
        radius = min(center_x, center_y) - 5
        
        # Draw horizon
        painter.setPen(QPen(QColor(100, 150, 255), 2))
        painter.setBrush(QBrush(QColor(100, 150, 255, 50)))
        
        # Rotate painter berdasarkan roll
        painter.save()
        painter.translate(center_x, center_y)
        painter.rotate(math.degrees(self.roll))
        
        # Draw horizon line
        horizon_y = -self.pitch * radius / (math.pi / 2)  # Scale pitch
        painter.drawLine(-radius, horizon_y, radius, horizon_y)
        
        painter.restore()
        
        # Draw aircraft symbol (fixed)
        painter.setPen(QPen(QColor(255, 255, 0), 3))
        painter.drawLine(center_x - 20, center_y, center_x + 20, center_y)
        painter.drawLine(center_x, center_y - 10, center_x, center_y + 10)
        
        # Draw yaw indicator
        painter.setPen(QPen(QColor(255, 100, 100), 2))
        yaw_x = center_x + (radius - 10) * math.sin(self.yaw)
        yaw_y = center_y - (radius - 10) * math.cos(self.yaw)
        painter.drawLine(center_x, center_y, yaw_x, yaw_y)