import threading
import time
import os
import sys
# Add the cv_custom folder to DLL search
cv_folder = os.path.join(os.getcwd(), "ui\\widgets\\cv_custom")
os.add_dll_directory(cv_folder)
sys.path.insert(0, cv_folder)
import cv_custom as cv3
import numpy as np

# Optional: suppress GTK / pygobject warnings
os.environ["GI_TYPELIB_PATH"] = ""
os.environ["PYGOBJECT_WARNINGS"] = "0"

class RTSPCamera:
    def __init__(self, rtsp_url, width_scale=0.5, height_scale=0.5):
        self.rtsp_url = rtsp_url
        self.width_scale = width_scale
        self.height_scale = height_scale
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.cap = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def update(self):
        gst_pipeline = (
            f'rtspsrc location={self.rtsp_url} latency=0 drop-on-latency=true ! '
            f'rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! '
            f'video/x-raw,format=BGR ! appsink drop=true sync=false max-buffers=1'
        )

        self.cap = cv3.VideoCapture(gst_pipeline, cv3.CAP_GSTREAMER)

        if not self.cap.isOpened():
            print("❌ Failed to open RTSP stream with GStreamer")
            self.running = False
            return

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]
            frame = cv3.resize(frame, (int(w * self.width_scale), int(h * self.height_scale)))

            with self.lock:
                self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

if __name__ == "__main__":
    print("Custom OpenCV version:", cv3.__version__)
    print("GStreamer enabled:", "GStreamer:                   YES" in cv3.getBuildInformation())

    rtsp_url = "rtsp://192.168.1.99:1234"  # <-- replace with your stream
    cam = RTSPCamera(rtsp_url, width_scale=0.5, height_scale=0.5)
    cam.start()

    try:
        while True:
            frame = cam.get_frame()
            if frame is not None:
                cv3.imshow("RTSP Viewer - GStreamer", frame)

            if cv3.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.001)

    finally:
        cam.stop()
        cv3.destroyAllWindows()
