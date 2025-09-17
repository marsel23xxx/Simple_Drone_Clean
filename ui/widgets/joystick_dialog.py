"""
Joystick Dialog for Orientation Control
Professional joystick interface for precise orientation editing
"""

import math
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont


class JoystickDialog(QDialog):
    """Dialog with joystick control for orientation editing."""
    
    def __init__(self, current_orientation=0.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Orientation")
        self.setFixedSize(300, 350)
        self.setModal(True)
        
        self.orientation = current_orientation
        self.joystick_center = (150, 150)
        self.joystick_radius = 80
        self.knob_pos = self.joystick_center
        self.dragging = False
        
        # Set initial knob position based on current orientation
        self.set_knob_from_orientation()
        
        self.setup_ui()
        self.apply_styling()
    
    def setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Orientation display
        self.orientation_label = QLabel(f"Orientation: {self.orientation:.3f} rad")
        self.orientation_label.setAlignment(Qt.AlignCenter)
        self.orientation_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(self.orientation_label)
        
        # Joystick area (will be drawn in paintEvent)
        layout.addStretch()
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def apply_styling(self):
        """Apply modern dark styling to the dialog."""
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: white;
                border-radius: 10px;
            }
            
            QLabel {
                color: white;
                background-color: transparent;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #4CAF50;
                margin: 5px;
            }
            
            QDialogButtonBox {
                background-color: transparent;
            }
            
            QPushButton {
                background-color: #4CAF50;
                border: 2px solid #66BB6A;
                border-radius: 8px;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                min-width: 80px;
            }
            
            QPushButton:hover {
                background-color: #66BB6A;
                border: 2px solid #81C784;
            }
            
            QPushButton:pressed {
                background-color: #388E3C;
                border: 2px solid #4CAF50;
            }
        """)
    
    def set_knob_from_orientation(self):
        """Set knob position based on orientation value."""
        # Up is 0, anticlockwise positive
        angle = -self.orientation
        x = self.joystick_center[0] + self.joystick_radius * 0.8 * math.sin(angle)
        y = self.joystick_center[1] + self.joystick_radius * 0.8 * math.cos(angle)
        self.knob_pos = (x, y)
    
    def update_orientation_from_knob(self):
        """Update orientation based on knob position."""
        dx = self.knob_pos[0] - self.joystick_center[0]
        dy = self.knob_pos[1] - self.joystick_center[1]
        
        if dx == 0 and dy == 0:
            self.orientation = 0.0
        else:
            # Calculate angle: up is 0, anticlockwise positive
            angle = math.atan2(dx, -dy)  # atan2(x, y) because up is 0
            self.orientation = -angle
        
        self.orientation_label.setText(f"Orientation: {self.orientation:.3f} rad")
    
    def paintEvent(self, event):
        """Draw the joystick interface."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw joystick base
        painter.setBrush(QBrush(QColor(60, 60, 60)))
        painter.setPen(QPen(QColor(100, 100, 100), 3))
        painter.drawEllipse(
            self.joystick_center[0] - self.joystick_radius,
            self.joystick_center[1] - self.joystick_radius,
            self.joystick_radius * 2, 
            self.joystick_radius * 2
        )
        
        # Draw inner circle
        inner_radius = self.joystick_radius - 10
        painter.setBrush(QBrush(QColor(40, 40, 40)))
        painter.setPen(QPen(QColor(80, 80, 80), 2))
        painter.drawEllipse(
            self.joystick_center[0] - inner_radius,
            self.joystick_center[1] - inner_radius,
            inner_radius * 2, 
            inner_radius * 2
        )
        
        # Draw direction indicators with modern styling
        painter.setPen(QPen(QColor(200, 200, 200), 2))
        painter.setFont(QFont("Arial", 11, QFont.Bold))
        
        # North (0°)
        painter.drawText(
            self.joystick_center[0] - 8, 
            self.joystick_center[1] - self.joystick_radius - 15, 
            "0°"
        )
        
        # East (90°)
        painter.drawText(
            self.joystick_center[0] + self.joystick_radius + 15, 
            self.joystick_center[1] + 5, 
            "90°"
        )
        
        # South (180°)
        painter.drawText(
            self.joystick_center[0] - 12, 
            self.joystick_center[1] + self.joystick_radius + 25, 
            "180°"
        )
        
        # West (-90°)
        painter.drawText(
            self.joystick_center[0] - self.joystick_radius - 35, 
            self.joystick_center[1] + 5, 
            "-90°"
        )
        
        # Draw direction lines
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        line_length = self.joystick_radius - 20
        
        # Vertical line
        painter.drawLine(
            self.joystick_center[0], 
            self.joystick_center[1] - line_length,
            self.joystick_center[0], 
            self.joystick_center[1] + line_length
        )
        
        # Horizontal line
        painter.drawLine(
            self.joystick_center[0] - line_length, 
            self.joystick_center[1],
            self.joystick_center[0] + line_length, 
            self.joystick_center[1]
        )
        
        # Draw orientation indicator line
        painter.setPen(QPen(QColor(76, 175, 80), 3))
        angle = -self.orientation
        line_end_x = self.joystick_center[0] + (self.joystick_radius - 25) * math.sin(angle)
        line_end_y = self.joystick_center[1] + (self.joystick_radius - 25) * math.cos(angle)
        
        painter.drawLine(
            self.joystick_center[0], 
            self.joystick_center[1],
            int(line_end_x), 
            int(line_end_y)
        )
        
        # Draw knob with enhanced appearance
        knob_radius = 15
        
        # Knob shadow
        painter.setBrush(QBrush(QColor(0, 0, 0, 100)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(
            int(self.knob_pos[0] - knob_radius + 2),
            int(self.knob_pos[1] - knob_radius + 2),
            knob_radius * 2, 
            knob_radius * 2
        )
        
        # Knob base
        painter.setBrush(QBrush(QColor(76, 175, 80)))
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawEllipse(
            int(self.knob_pos[0] - knob_radius),
            int(self.knob_pos[1] - knob_radius),
            knob_radius * 2, 
            knob_radius * 2
        )
        
        # Knob highlight
        painter.setBrush(QBrush(QColor(129, 199, 132)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(
            int(self.knob_pos[0] - knob_radius//2),
            int(self.knob_pos[1] - knob_radius//2),
            knob_radius, 
            knob_radius
        )
        
        # Draw current orientation value in the center
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        orientation_text = f"{math.degrees(self.orientation):.1f}°"
        text_rect = painter.fontMetrics().boundingRect(orientation_text)
        painter.drawText(
            self.joystick_center[0] - text_rect.width()//2,
            self.joystick_center[1] + text_rect.height()//2,
            orientation_text
        )
    
    def mousePressEvent(self, event):
        """Handle mouse press on joystick."""
        if event.button() == Qt.LeftButton:
            knob_radius = 15
            dx = event.x() - self.knob_pos[0]
            dy = event.y() - self.knob_pos[1]
            
            # Check if click is on knob
            if dx*dx + dy*dy <= knob_radius*knob_radius:
                self.dragging = True
                self.setCursor(Qt.ClosedHandCursor)
            else:
                # Click outside knob - move knob to click position
                self.update_knob_position(event.x(), event.y())
    
    def mouseMoveEvent(self, event):
        """Handle knob dragging."""
        if self.dragging or event.buttons() & Qt.LeftButton:
            self.update_knob_position(event.x(), event.y())
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.setCursor(Qt.ArrowCursor)
    
    def update_knob_position(self, x, y):
        """Update knob position with constraints."""
        # Constrain knob to joystick area
        dx = x - self.joystick_center[0]
        dy = y - self.joystick_center[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        max_distance = self.joystick_radius - 20  # Keep knob inside the circle
        
        if distance <= max_distance:
            self.knob_pos = (x, y)
        else:
            # Constrain to circle edge
            factor = max_distance / distance
            self.knob_pos = (
                self.joystick_center[0] + dx * factor,
                self.joystick_center[1] + dy * factor
            )
        
        self.update_orientation_from_knob()
        self.update()
    
    def keyPressEvent(self, event):
        """Handle keyboard input for fine adjustment."""
        step = 0.1  # Radians
        
        if event.key() == Qt.Key_Left:
            self.orientation -= step
        elif event.key() == Qt.Key_Right:
            self.orientation += step
        elif event.key() == Qt.Key_Up:
            self.orientation = 0.0
        elif event.key() == Qt.Key_Down:
            self.orientation = math.pi
        else:
            super().keyPressEvent(event)
            return
        
        # Normalize orientation to [-π, π]
        while self.orientation > math.pi:
            self.orientation -= 2 * math.pi
        while self.orientation < -math.pi:
            self.orientation += 2 * math.pi
        
        self.set_knob_from_orientation()
        self.orientation_label.setText(f"Orientation: {self.orientation:.3f} rad")
        self.update()