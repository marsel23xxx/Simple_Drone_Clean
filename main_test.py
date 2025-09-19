#!/usr/bin/env python3
"""
Main entry point for Drone Control Center Application
Updated untuk menggunakan UI design yang sudah ada
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

from ui.main_window import DroneControlMainWindow
from ui.styles.dark_theme import apply_dark_theme
from config.settings import APP_CONFIG

def setup_application():
    """Setup the QApplication with proper configurations."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName(APP_CONFIG['app_name'])
    app.setApplicationVersion(APP_CONFIG['app_version'])
    app.setOrganizationName(APP_CONFIG['organization'])
    
    # Set application icon if available
    icon_path = project_root / "assets" / "icons" / "app_icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    
    # Apply dark theme
    apply_dark_theme(app)
    
    # Enable high DPI scaling
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Set environment variables for better rendering
    if 'QT_AUTO_SCREEN_SCALE_FACTOR' not in os.environ:
        os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
    
    return app

def main():
    """Main application entry point."""
    try:
        # Create application
        app = setup_application()
        
        # Create and show main window
        main_window = DroneControlMainWindow()
        main_window.show()
        
        # Set window title
        main_window.setWindowTitle(APP_CONFIG['window_title'])
        
        # Start application event loop
        return app.exec_()
        
    except Exception as e:
        print(f"Application startup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())