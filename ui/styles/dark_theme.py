# ui/dark_theme.py

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor

def apply_dark_theme(app):
    """Apply dark theme to the entire application."""
    app.setStyle('Fusion')
    
    # Create dark palette
    dark_palette = QPalette()
    
    # Window colors
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    
    # Base colors
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    
    # Text colors
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    
    # Highlight colors
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    
    app.setPalette(dark_palette)
    
    # Apply global stylesheet
    app.setStyleSheet(get_global_stylesheet())

def get_global_stylesheet():
    """Get the global stylesheet for the application."""
    return """
    /* Global Application Styles */
    QMainWindow {
        background-color: rgb(85, 85, 85);
        color: white;
    }
    
    /* Frame Styles */
    QFrame {
        border-radius: 10px;
        background-color: #4d4d4d;
    }
    
    QFrame[frameStyle="panel"] {
        border: 1px solid #3c3c3c;
    }
    
    /* Button Styles */
    QPushButton {
        background-color: rgb(65, 65, 65);
        border: 2px solid black;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        padding: 6px 12px;
        text-align: center;
        min-height: 15px;
    }
    
    QPushButton:hover {
        background-color: rgb(90, 90, 90);
        border: 2px solid rgb(120, 120, 120);
        color: #e0e0e0;
    }
    
    QPushButton:pressed {
        background-color: rgb(40, 40, 40);
        border: 2px solid rgb(100, 100, 100);
        color: #cccccc;
    }
    
    QPushButton:disabled {
        background-color: rgb(30, 30, 30);
        border: 2px solid rgb(50, 50, 50);
        color: rgb(100, 100, 100);
    }
    
    /* Label Styles */
    QLabel {
        color: rgb(209, 207, 207);
        border: 0px;
    }
    
    QLabel[styleClass="title"] {
        font-weight: bold;
        font-size: 14px;
        color: rgb(156, 156, 156);
    }
    
    QLabel[styleClass="data"] {
        color: white;
        font-weight: bold;
    }
    
    /* Progress Bar Styles */
    QProgressBar {
        border: 2px solid #555;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        background-color: #2b2b2b;
        color: white;
    }
    
    QProgressBar::chunk {
        background-color: #3ddb55;
        border-radius: 9px;
    }
    
    QProgressBar::chunk[value="0"] {
        background-color: #e63946;
    }
    
    QProgressBar::chunk[value="25"] {
        background-color: #f77f00;
    }
    
    QProgressBar::chunk[value="50"] {
        background-color: #fcbf49;
    }
    
    /* CheckBox Styles */
    QCheckBox {
        spacing: 8px;
        font-size: 11px;
        color: #f0f0f0;
        font-weight: bold;
        border: none;
    }
    
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border-radius: 4px;
        border: 2px solid #4CAF50;
        background-color: #2b2b2b;
    }
    
    QCheckBox::indicator:hover {
        border: 2px solid #66BB6A;
        background-color: #333333;
    }
    
    QCheckBox::indicator:checked {
        background-color: #4CAF50;
        border: 2px solid #66BB6A;
    }
    
    QCheckBox::indicator:disabled {
        background-color: #555555;
        border: 2px solid #777777;
    }
    
    /* Table Styles */
    QTableWidget {
        background-color: #1e1e1e;
        alternate-background-color: #2a2a2a;
        gridline-color: #3c3c3c;
        border: 1px solid #3c3c3c;
        border-radius: 6px;
        font-size: 14px;
        color: #f0f0f0;
        selection-background-color: #e63946;
        selection-color: white;
    }
    
    QHeaderView::section {
        background-color: #2f2f2f;
        color: #ffffff;
        padding: 6px;
        font-weight: bold;
        border: 1px solid #3c3c3c;
        border-left: none;
    }
    
    QTableCornerButton::section {
        background-color: #2f2f2f;
        border: 1px solid #3c3c3c;
    }
    
    /* Slider Styles */
    QSlider {
        background-color: #4d4d4d;
        border-radius: 10px;
    }
    
    QSlider::groove:vertical {
        background: transparent;
        width: 6px;
        margin: 0 20px;
    }
    
    QSlider::sub-page:vertical, 
    QSlider::add-page:vertical {
        background: transparent;
    }
    
    QSlider::handle:vertical {
        background: #ff9933;
        border: 2px solid #cc6600;
        height: 28px;
        width: 136px;
        border-radius: 6px;
        margin: 30px -20px;
    }
    
    QSlider::handle:vertical:hover {
        background: #ffb366;
        border: 2px solid #ff6600;
    }
    
    /* SpinBox Styles */
    QDoubleSpinBox {
        background-color: #2b2b2b;
        border: 1px solid #4CAF50;
        border-radius: 6px;
        padding: 4px 28px 4px 8px;
        color: #f0f0f0;
        font-size: 13px;
        selection-background-color: #4CAF50;
    }
    
    QDoubleSpinBox:hover {
        border: 1px solid #66FF99;
    }
    
    QDoubleSpinBox:focus {
        border: 1px solid #00FFCC;
        background-color: #333333;
    }
    
    QDoubleSpinBox::up-button, 
    QDoubleSpinBox::down-button {
        background-color: #3c3c3c;
        border: none;
        width: 20px;
        border-radius: 4px;
    }
    
    QDoubleSpinBox::up-button:hover, 
    QDoubleSpinBox::down-button:hover {
        background-color: #4CAF50;
    }
    
    /* Scrollbar Styles */
    QScrollBar:vertical {
        border: none;
        background: #2b2b2b;
        width: 10px;
        margin: 0px;
        border-radius: 4px;
    }
    
    QScrollBar::handle:vertical {
        background: #444;
        min-height: 20px;
        border-radius: 4px;
    }
    
    QScrollBar::handle:vertical:hover {
        background: #666;
    }
    
    QScrollBar:horizontal {
        border: none;
        background: #2b2b2b;
        height: 10px;
        margin: 0px;
        border-radius: 4px;
    }
    
    QScrollBar::handle:horizontal {
        background: #444;
        min-width: 20px;
        border-radius: 4px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background: #666;
    }
    
    /* TextBrowser Styles */
    QTextBrowser {
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #3c3c3c;
        border-radius: 6px;
        padding: 5px;
        font-family: "Consolas", "Monaco", monospace;
    }
    
    /* Dial Styles */
    QDial {
        background-color: #3c3c3c;
        border-radius: 75px;
    }
    
    /* Status Indicator Styles */
    .status-online {    
        background-color: rgb(0, 255, 0);
        border: none;
        min-width: 10px;
        min-height: 10px;
        max-width: 10px;  
        max-height: 10px; 
        border-radius: 5px;
    }
    
    .status-offline {
        background-color: rgb(255, 0, 0);
        border: none;
        min-width: 10px;
        min-height: 10px;
        max-width: 10px;  
        max-height: 10px; 
        border-radius: 5px;
    }
    
    .status-warning {
        background-color: rgb(255, 165, 0);
        border: none;
        min-width: 10px;
        min-height: 10px;
        max-width: 10px;  
        max-height: 10px; 
        border-radius: 5px;
    }
    """

def get_emergency_button_style():
    """Get emergency button specific styling."""
    return """
    QPushButton {
        border: none;
        border-radius: 8px;
        background-color: transparent;
        min-width: 20px;
        min-height: 20px;
    }
    
    QPushButton:hover {
        background-color: rgba(0, 0, 0, 50);
        border-radius: 8px;
    }
    
    QPushButton:pressed {
        background-color: rgba(0, 0, 0, 100);
        border-radius: 8px;
    }
    """

def get_connection_button_style():
    """Get connection button specific styling."""
    return """
    QPushButton {
        background-color: rgb(65, 65, 65);
        border: 2px solid black;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        padding: 6px 12px;
        text-align: right;
    }
    """