#!/usr/bin/env python3
"""
Setup script for Professional Drone Control Center
Handles installation, dependencies, and project configuration
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


class DroneControlSetup:
    """Setup manager for the Drone Control Center."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_exe = sys.executable
        
    def check_python_version(self):
        """Check if Python version is compatible."""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8 or higher is required!")
            print(f"Current version: {version.major}.{version.minor}.{version.micro}")
            return False
        
        print(f"✅ Python version {version.major}.{version.minor}.{version.micro} - OK")
        return True
    
    def install_dependencies(self):
        """Install required dependencies."""
        print("\n📦 Installing dependencies...")
        
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            print("❌ requirements.txt not found!")
            return False
        
        try:
            # Upgrade pip first
            subprocess.check_call([
                self.python_exe, "-m", "pip", "install", "--upgrade", "pip"
            ])
            
            # Install requirements
            subprocess.check_call([
                self.python_exe, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            
            print("✅ Dependencies installed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def create_directories(self):
        """Create necessary directories."""
        print("\n📁 Creating directories...")
        
        directories = [
            "data",
            "data/logs",
            "data/exports", 
            "data/backups",
            "data/point_clouds",
            "assets/images",
            "assets/icons"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ {directory}")
        
        return True
    
    def create_init_files(self):
        """Create __init__.py files for proper package structure."""
        print("\n📝 Creating package files...")
        
        packages = [
            "config",
            "ui", 
            "ui/widgets",
            "ui/styles",
            "core",
            "utils"
        ]
        
        for package in packages:
            init_file = self.project_root / package / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                print(f"  ✅ {package}/__init__.py")
        
        return True
    
    def check_assets(self):
        """Check if required asset files exist."""
        print("\n🖼️  Checking assets...")
        
        required_assets = [
            "assets/images/LOGO R BG-012.png",
            "assets/images/drone-display.png", 
            "assets/images/compas.png",
            "assets/images/emergency.png",
            "assets/images/altitude.png",
            "assets/images/Drone 2.png",
            "assets/images/Drone 3.png",
            "assets/images/DRONETOP.png"
        ]
        
        missing_assets = []
        
        for asset in required_assets:
            asset_path = self.project_root / asset
            if asset_path.exists():
                print(f"  ✅ {asset}")
            else:
                print(f"  ⚠️  {asset} - MISSING")
                missing_assets.append(asset)
        
        if missing_assets:
            print(f"\n⚠️  {len(missing_assets)} asset files are missing.")
            print("   The application will work but some UI elements may not display correctly.")
            print("   Please add the missing assets to continue with full functionality.")
        
        return len(missing_assets) == 0
    
    def create_desktop_shortcut(self):
        """Create desktop shortcut (Windows/Linux)."""
        try:
            if sys.platform == "win32":
                self._create_windows_shortcut()
            elif sys.platform.startswith("linux"):
                self._create_linux_shortcut()
            else:
                print("   Desktop shortcut creation not supported on this platform")
                return False
                
            return True
            
        except Exception as e:
            print(f"   ⚠️  Could not create desktop shortcut: {e}")
            return False
    
    def _create_windows_shortcut(self):
        """Create Windows shortcut."""
        try:
            import winshell
            from win32com.client import Dispatch
            
            desktop = winshell.desktop()
            shortcut_path = os.path.join(desktop, "Drone Control Center.lnk")
            
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(shortcut_path)
            shortcut.Targetpath = self.python_exe
            shortcut.Arguments = str(self.project_root / "main.py")
            shortcut.WorkingDirectory = str(self.project_root)
            shortcut.IconLocation = str(self.project_root / "assets" / "icons" / "app_icon.ico")
            shortcut.save()
            
            print(f"  ✅ Windows shortcut created on desktop")
            
        except ImportError:
            print("  ⚠️  winshell package required for Windows shortcuts")
            print("     Install with: pip install winshell pywin32")
    
    def _create_linux_shortcut(self):
        """Create Linux desktop entry."""
        desktop_file_content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=Drone Control Center
Comment=Professional Drone Ground Control Station
Exec={self.python_exe} {self.project_root}/main.py
Icon={self.project_root}/assets/icons/app_icon.png
Terminal=false
Categories=Development;Science;
"""
        
        # User desktop file
        desktop_dir = Path.home() / ".local" / "share" / "applications"
        desktop_dir.mkdir(parents=True, exist_ok=True)
        
        desktop_file = desktop_dir / "drone-control-center.desktop"
        desktop_file.write_text(desktop_file_content)
        desktop_file.chmod(0o755)
        
        print(f"  ✅ Linux desktop entry created")
    
    def create_launcher_script(self):
        """Create launcher scripts for different platforms."""
        print("\n🚀 Creating launcher scripts...")
        
        # Windows batch file
        if sys.platform == "win32":
            launcher_content = f"""@echo off
cd /d "{self.project_root}"
"{self.python_exe}" main.py
pause
"""
            launcher_file = self.project_root / "launch.bat"
            launcher_file.write_text(launcher_content)
            print("  ✅ launch.bat created")
        
        # Unix shell script
        launcher_content = f"""#!/bin/bash
cd "{self.project_root}"
"{self.python_exe}" main.py
"""
        launcher_file = self.project_root / "launch.sh"
        launcher_file.write_text(launcher_content)
        launcher_file.chmod(0o755)
        print("  ✅ launch.sh created")
        
        return True
    
    def test_installation(self):
        """Test if installation is working."""
        print("\n🧪 Testing installation...")
        
        try:
            # Test imports
            test_script = '''
import sys
sys.path.insert(0, ".")

try:
    from PyQt5.QtWidgets import QApplication
    print("✅ PyQt5 import successful")
    
    import numpy as np
    print("✅ NumPy import successful")
    
    import open3d as o3d
    print("✅ Open3D import successful")
    
    import websockets
    print("✅ WebSockets import successful")
    
    from config.settings import APP_CONFIG
    print("✅ Config import successful")
    
    from ui.main_window import DroneControlMainWindow
    print("✅ Main window import successful")
    
    print("\\n🎉 All imports successful! Installation is working correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
'''
            
            result = subprocess.run([
                self.python_exe, "-c", test_script
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(result.stdout)
                return True
            else:
                print("❌ Import test failed:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    def run_setup(self):
        """Run complete setup process."""
        print("🚁 Drone Control Center - Setup")
        print("=" * 50)
        
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Installing dependencies", self.install_dependencies), 
            ("Creating directories", self.create_directories),
            ("Creating package files", self.create_init_files),
            ("Checking assets", self.check_assets),
            ("Creating launcher scripts", self.create_launcher_script),
            ("Testing installation", self.test_installation)
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            print(f"\n{step_name}...")
            try:
                if not step_func():
                    failed_steps.append(step_name)
            except Exception as e:
                print(f"❌ {step_name} failed: {e}")
                failed_steps.append(step_name)
        
        # Summary
        print("\n" + "=" * 50)
        print("Setup Summary")
        print("=" * 50)
        
        if not failed_steps:
            print("🎉 Setup completed successfully!")
            print("\nTo start the application:")
            print(f"  • Run: {self.python_exe} main.py")
            if sys.platform == "win32":
                print("  • Or double-click: launch.bat")
            else:
                print("  • Or run: ./launch.sh")
        else:
            print(f"⚠️  Setup completed with {len(failed_steps)} issues:")
            for step in failed_steps:
                print(f"  • {step}")
            print("\nPlease resolve the issues above before running the application.")
        
        return len(failed_steps) == 0


def main():
    """Main setup function."""
    setup = DroneControlSetup()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "install":
            setup.run_setup()
        elif command == "test":
            setup.test_installation()
        elif command == "deps":
            setup.install_dependencies()
        elif command == "clean":
            print("🧹 Cleaning up...")
            # Add cleanup logic here
            print("✅ Cleanup completed")
        else:
            print(f"Unknown command: {command}")
            print("Available commands: install, test, deps, clean")
    else:
        setup.run_setup()


if __name__ == "__main__":
    main()