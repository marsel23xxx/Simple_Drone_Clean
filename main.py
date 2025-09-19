#!/usr/bin/env python3
"""
Main entry point for Drone Control Center Application
Optimized startup sequence untuk UDP telemetry integration
"""

import sys
import os
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QPixmap

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

def create_splash_screen():
    """Create and show splash screen during startup."""
    try:
        # Try to load splash screen image
        splash_path = project_root / "assets" / "splash" / "splash.png"
        if splash_path.exists():
            pixmap = QPixmap(str(splash_path))
        else:
            # Create simple splash screen
            pixmap = QPixmap(400, 300)
            pixmap.fill(Qt.darkGray)
        
        splash = QSplashScreen(pixmap)
        splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.SplashScreen)
        splash.show()
        
        return splash
    except Exception as e:
        print(f"Could not create splash screen: {e}")
        return None

def update_splash_message(splash, message):
    """Update splash screen message."""
    if splash:
        splash.showMessage(message, Qt.AlignBottom | Qt.AlignCenter, Qt.white)
        QApplication.processEvents()

def startup_with_splash(app):
    """Startup sequence with splash screen feedback."""
    
    # Create splash screen
    splash = create_splash_screen()
    
    try:
        # Step 1: Initialize application
        update_splash_message(splash, "Initializing Drone Control Center...")
        time.sleep(0.5)
        
        # Step 2: Load UI components
        update_splash_message(splash, "Loading UI components...")
        main_window = DroneControlMainWindow()
        time.sleep(0.3)
        
        # Step 3: Check UDP telemetry setup
        update_splash_message(splash, "Setting up UDP telemetry...")
        if main_window.drone_parser:
            update_splash_message(splash, f"UDP telemetry ready on port {main_window.drone_parser.port}")
        else:
            update_splash_message(splash, "UDP telemetry unavailable (install scapy)")
        time.sleep(0.5)
        
        # Step 4: Show main window
        update_splash_message(splash, "Starting main window...")
        main_window.setWindowTitle(APP_CONFIG['window_title'])
        main_window.show()
        
        # Step 5: Wait for services to start
        update_splash_message(splash, "Starting communication services...")
        
        # Use timer to close splash after services start
        def close_splash_and_log():
            if splash:
                splash.close()
            # Log startup status
            if main_window.drone_parser and hasattr(main_window.drone_parser, 'is_running'):
                if main_window.drone_parser.is_running:
                    main_window.log_debug("✓ UDP telemetry active - waiting for drone data...")
                else:
                    main_window.log_debug("⚠ UDP telemetry not started - check DroneParser")
            else:
                main_window.log_debug("⚠ UDP telemetry disabled - install scapy for telemetry")
        
        # Close splash after 2 seconds
        QTimer.singleShot(2000, close_splash_and_log)
        
        return main_window
        
    except Exception as e:
        if splash:
            splash.close()
        raise e

def startup_without_splash(app):
    """Simple startup without splash screen."""
    # Create and show main window
    main_window = DroneControlMainWindow()
    main_window.setWindowTitle(APP_CONFIG['window_title'])
    main_window.show()
    
    # Log startup status after brief delay
    def log_startup_status():
        if main_window.drone_parser and hasattr(main_window.drone_parser, 'is_running'):
            if main_window.drone_parser.is_running:
                main_window.log_debug("✓ UDP telemetry active - waiting for drone data...")
                print("UDP telemetry started successfully")
            else:
                main_window.log_debug("⚠ UDP telemetry not started - check DroneParser")
                print("Warning: UDP telemetry not started")
        else:
            main_window.log_debug("⚠ UDP telemetry disabled - install scapy for telemetry")
            print("Warning: UDP telemetry disabled")
    
    QTimer.singleShot(1000, log_startup_status)
    
    return main_window

def main():
    """Main application entry point with optimized startup."""
    try:
        print("Starting Drone Control Center...")
        
        # Create application
        app = setup_application()
        
        # Choose startup method (with or without splash)
        use_splash = True  # Set to False to disable splash screen
        
        if use_splash:
            main_window = startup_with_splash(app)
        else:
            main_window = startup_without_splash(app)
        
        # Print startup summary
        print(f"Application: {APP_CONFIG['app_name']} v{APP_CONFIG['app_version']}")
        print(f"Main window created successfully")
        
        # Check critical components
        if hasattr(main_window, 'drone_parser') and main_window.drone_parser:
            print(f"UDP Telemetry: Listening on port {main_window.drone_parser.port}")
        else:
            print("UDP Telemetry: Disabled (scapy not available)")
            
        if hasattr(main_window, 'telemetry_handler') and main_window.telemetry_handler:
            print("Telemetry Handler: Ready")
        else:
            print("Telemetry Handler: Not available")
        
        print("Drone Control Center startup complete")
        print("=" * 50)
        
        # Start application event loop
        return app.exec_()
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user (Ctrl+C)")
        return 0
        
    except Exception as e:
        print(f"Application startup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

def check_dependencies():
    """Check if critical dependencies are available."""
    print("Checking dependencies...")
    
    missing_deps = []
    
    # Check PyQt5
    try:
        import PyQt5
        print("✓ PyQt5 available")
    except ImportError:
        missing_deps.append("PyQt5")
        print("✗ PyQt5 not found")
    
    # Check scapy for UDP telemetry
    try:
        import scapy
        print("✓ scapy available (UDP telemetry enabled)")
    except ImportError:
        print("⚠ scapy not found (UDP telemetry disabled)")
        print("  Install with: pip install scapy")
    
    # Check other optional dependencies
    optional_deps = {
        'open3d': 'Point cloud processing',
        'numpy': 'Numerical operations', 
        'websocket': 'WebSocket communication'
    }
    
    for dep, description in optional_deps.items():
        try:
            __import__(dep)
            print(f"✓ {dep} available ({description})")
        except ImportError:
            print(f"⚠ {dep} not found ({description} disabled)")
    
    if missing_deps:
        print(f"\nCritical dependencies missing: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False
    
    print("Dependencies check complete")
    return True

if __name__ == "__main__":
    # Check dependencies first
    if not check_dependencies():
        print("Please install missing dependencies before running the application")
        sys.exit(1)
    
    # Run main application
    sys.exit(main())