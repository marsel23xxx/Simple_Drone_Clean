#!/usr/bin/env python3
"""
Application Launcher for Drone Control Center
Provides startup checks and error handling before launching the main application
"""

import sys
import os
import traceback
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_python_version():
    """Check if Python version meets requirements."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"Error: Python 3.8+ required. Current: {version.major}.{version.minor}")
        return False
    return True


def check_dependencies():
    """Check if required dependencies are available."""
    required_modules = [
        ('PyQt5', 'PyQt5.QtWidgets'),
        ('numpy', 'numpy'),
        ('open3d', 'open3d'), 
        ('websockets', 'websockets')
    ]
    
    missing = []
    
    for name, import_path in required_modules:
        try:
            __import__(import_path)
        except ImportError:
            missing.append(name)
    
    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    return True


def check_project_structure():
    """Verify project structure is complete."""
    required_dirs = [
        'config',
        'ui',
        'ui/widgets', 
        'ui/styles',
        'core',
        'utils',
        'data'
    ]
    
    required_files = [
        'main.py',
        'config/settings.py',
        'ui/main_window.py'
    ]
    
    missing_dirs = []
    missing_files = []
    
    for directory in required_dirs:
        if not (project_root / directory).exists():
            missing_dirs.append(directory)
    
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)
    
    if missing_dirs or missing_files:
        print("Project structure incomplete:")
        for item in missing_dirs:
            print(f"  Missing directory: {item}")
        for item in missing_files:
            print(f"  Missing file: {item}")
        return False
    
    return True


def setup_environment():
    """Setup environment variables and paths."""
    # Set QT scale factor for high DPI displays
    if 'QT_AUTO_SCREEN_SCALE_FACTOR' not in os.environ:
        os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
    
    # Disable OpenGL warnings in some environments
    if 'QT_OPENGL' not in os.environ:
        os.environ['QT_OPENGL'] = 'software'


def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler."""
    if issubclass(exc_type, KeyboardInterrupt):
        print("\nApplication interrupted by user")
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    print("An unexpected error occurred:")
    print("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
    
    # Try to show error in GUI if PyQt5 is available
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox
        
        if QApplication.instance() is None:
            app = QApplication([])
        
        error_msg = f"An unexpected error occurred:\n\n{''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))}"
        
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Drone Control Center - Error")
        msg_box.setText("An unexpected error occurred")
        msg_box.setDetailedText(error_msg)
        msg_box.exec_()
        
    except ImportError:
        pass  # PyQt5 not available, error already printed


def main():
    """Main launcher function."""
    print("Drone Control Center - Starting...")
    
    # Install global exception handler
    sys.excepthook = handle_exception
    
    # Run startup checks
    checks = [
        ("Python version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project structure", check_project_structure)
    ]
    
    for check_name, check_func in checks:
        try:
            if not check_func():
                print(f"Startup check failed: {check_name}")
                input("Press Enter to exit...")
                sys.exit(1)
        except Exception as e:
            print(f"Error during {check_name} check: {e}")
            input("Press Enter to exit...")
            sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Import and launch main application
    try:
        from main import main as app_main
        print("Starting application...")
        return app_main()
        
    except ImportError as e:
        print(f"Failed to import main application: {e}")
        print("Make sure main.py exists and is properly configured")
        input("Press Enter to exit...")
        return 1
        
    except Exception as e:
        print(f"Failed to start application: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"Launcher error: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)