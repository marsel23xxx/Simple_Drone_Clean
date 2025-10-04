# ui/widgets/cv_custom/__init__.py
import sys
import os

# Ensure DLLs are found
dll_path = os.path.dirname(__file__)
if dll_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = dll_path + os.pathsep + os.environ.get("PATH", "")
print(f"Custom OpenCV DLL path added: {dll_path}")

# TAMBAHAN: Add to sys.path untuk memastikan module bisa di-import
if dll_path not in sys.path:
    sys.path.insert(0, dll_path)

# Import the original .pyd as cv_custom
import cv2 as _cv2

# Export semua atribut dari cv2 ke namespace cv_custom
globals().update(_cv2.__dict__)

# TAMBAHAN: Export __version__ dan __all__ explicitly
__version__ = _cv2.__version__
__all__ = dir(_cv2)

# TAMBAHAN: Print konfirmasi
print(f"Custom OpenCV loaded successfully: version {__version__}")