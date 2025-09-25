import sys
import os

# Ensure DLLs are found
dll_path = os.path.dirname(__file__)
if dll_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = dll_path + os.pathsep + os.environ.get("PATH", "")
print(f"Custom OpenCV DLL path added: {dll_path}")
# Import the original .pyd as cv_custom
import cv2 as _cv2
globals().update(_cv2.__dict__)

