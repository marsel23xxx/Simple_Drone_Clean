# import threading
# import time
# import os
# import sys
# import socket
# import requests
# import numpy as np

# # ================== OpenCV import (preserve cv_custom if present) ==================
# cv_folder = os.path.join(os.getcwd(), "cv_custom")
# try:
#     os.add_dll_directory(cv_folder)  # Windows only; harmless on Linux in try/except
# except Exception:
#     pass
# sys.path.insert(0, cv_folder)
# os.environ.setdefault("GST_PLUGIN_PATH", cv_folder)

# try:
#     import cv_custom as cv3  # your custom OpenCV build
# except Exception:
#     import cv2 as cv3        # fallback to stock OpenCV if cv_custom unavailable

# # Quiet some GTK/pygobject warnings if present
# os.environ.setdefault("GI_TYPELIB_PATH", "")
# os.environ.setdefault("PYGOBJECT_WARNINGS", "0")


# # ========================= Helpers =========================

# def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
#     s = socket.socket()
#     s.settimeout(timeout)
#     try:
#         s.connect((host, port))
#         return True
#     except Exception:
#         return False
#     finally:
#         try:
#             s.close()
#         except Exception:
#             pass


# # ========================= RTSP via GStreamer =========================

# class RTSPCamera:
#     def __init__(self, rtsp_url: str, width_scale: float = 0.5, height_scale: float = 0.5, latency_ms: int = 0):
#         self.rtsp_url = rtsp_url
#         self.width_scale = width_scale
#         self.height_scale = height_scale
#         self.latency_ms = latency_ms
#         self.frame = None
#         self.running = False
#         self.lock = threading.Lock()
#         self.cap = None
#         self.thread = None

#     def start(self):
#         if self.running:
#             return
#         self.running = True
#         self.thread = threading.Thread(target=self._loop, daemon=True)
#         self.thread.start()

#     def stop(self):
#         self.running = False
#         if self.cap:
#             try:
#                 self.cap.release()
#             except Exception:
#                 pass
#         self.cap = None

#     def _open(self):
#         # Base rtspsrc (add protocols only if explicitly requested)
#         base = f"rtspsrc location={self.rtsp_url} latency={self.latency_ms} drop-on-latency=true"
#         proto = os.environ.get("RTSP_PROTO", "").lower()
#         if proto in ("udp", "tcp"):
#             base = base.replace("rtspsrc", f"rtspsrc protocols={proto}")
    
#         # Ordered decoder candidates (env first if set)
#         dec_env = os.environ.get("H264_DEC", "").strip()
#         decoders = ([dec_env] if dec_env else []) + [
#             "nvh264dec",        # NVIDIA (desktop)
#             "vaapih264dec",     # Intel iGPU (Linux)
#             "d3d11h264dec",     # Windows DX11
#             "nvv4l2decoder",    # NVIDIA Jetson
#             "avdec_h264",       # CPU fallback (always last)
#         ]
    
#         def build_pipeline(dec_name: str) -> str:
#             # Only avdec supports "max-threads" (avoid breaking other decoders)
#             if dec_name.startswith("avdec_"):
#                 dec_stage = f"{dec_name} max-threads=1"
#             else:
#                 dec_stage = dec_name
    
#             # Jetson prefers nvvidconv; others use videoconvert
#             convert = "nvvidconv" if dec_name == "nvv4l2decoder" else "videoconvert"
    
#             return (
#                 f"{base} ! "
#                 f"rtph264depay ! h264parse config-interval=1 ! "
#                 f"{dec_stage} ! {convert} ! video/x-raw,format=BGR ! "
#                 f"appsink drop=true sync=false max-buffers=1"
#             )
    
#         # Try decoders until one opens
#         tried = []
#         for dec in decoders:
#             if not dec:
#                 continue
#             pipe = build_pipeline(dec)
#             cap = cv3.VideoCapture(pipe, cv3.CAP_GSTREAMER)
#             if cap and cap.isOpened():
#                 # Optional: print which pipeline succeeded
#                 print(f"✅ RTSP using decoder: {dec} (proto={proto or 'default'})")
#                 return cap
#             tried.append(dec)
#             if cap:
#                 cap.release()
    
#         print(f"⚠ RTSP GStreamer failed with decoders: {tried}. Falling back to FFmpeg backend.")
#         # Last-ditch fallback (OpenCV FFmpeg). Latency may be higher but it’s better than failing.
#         return cv3.VideoCapture(self.rtsp_url)
    
    
    

#     def _loop(self):
#         while self.running:
#             self.cap = self._open()
#             if not self.cap or not self.cap.isOpened():
#                 print("❌ Failed to open RTSP stream (GStreamer). Retrying in 1s…")
#                 time.sleep(1)
#                 continue
#             while self.running:
#                 ret, frame = self.cap.read()
#                 if not ret:
#                     time.sleep(0.02)
#                     break
#                 h, w = frame.shape[:2]
#                 if self.width_scale != 1.0 or self.height_scale != 1.0:
#                     frame = cv3.resize(frame, (int(w * self.width_scale), int(h * self.height_scale)))
#                 with self.lock:
#                     self.frame = frame
#             try:
#                 self.cap.release()
#             except Exception:
#                 pass
#             self.cap = None

#     def get_frame(self):
#         with self.lock:
#             return None if self.frame is None else self.frame.copy()


# # ========================= HTTP MJPEG (Flask /video_feed) =========================

# class HTTPStreamCamera:
#     """
#     Robust MJPEG reader for Flask/OpenCV endpoints like
#     http://HOST:PORT/video_feed (multipart/x-mixed-replace; boundary=...)

#     Avoids OpenCV's VideoCapture for HTTP, parses the multipart stream directly
#     to minimize buffering and stutter.
#     """
#     def __init__(self, base_url: str, path: str = "/video_feed", width_scale: float = 0.5, height_scale: float = 0.5):
#         self.base_url = base_url.rstrip('/')
#         self.url = self.base_url + (path if path.startswith('/') else '/' + path)
#         self.width_scale = width_scale
#         self.height_scale = height_scale
#         self.frame = None
#         self.running = False
#         self.lock = threading.Lock()
#         self.thread = None

#     def start(self):
#         if self.running:
#             return
#         self.running = True
#         self.thread = threading.Thread(target=self._loop_mjpeg_manual, daemon=True)
#         self.thread.start()

#     def stop(self):
#         self.running = False

#     def _loop_mjpeg_manual(self):
#         session = requests.Session()
#         headers = {
#             "Accept": "multipart/x-mixed-replace, image/jpeg",
#             "Cache-Control": "no-cache",
#             "Connection": "keep-alive",
#             "Pragma": "no-cache",
#         }
#         while self.running:
#             try:
#                 with session.get(self.url, stream=True, timeout=(3.0, 10.0), headers=headers) as r:
#                     ctype = r.headers.get('Content-Type', '')
#                     boundary = None
#                     if 'multipart' in ctype.lower():
#                         for t in ctype.split(';'):
#                             t = t.strip()
#                             if t.lower().startswith('boundary='):
#                                 boundary = t.split('=', 1)[1].strip().strip('"')
#                                 break
#                     if not boundary:
#                         boundary = 'frame'  # common default in Flask examples
#                     bnd = ('--' + boundary).encode('utf-8')

#                     buf = b''
#                     for chunk in r.iter_content(chunk_size=2048):
#                         if not self.running:
#                             break
#                         if not chunk:
#                             continue
#                         buf += chunk

#                         while True:
#                             start = buf.find(bnd)
#                             if start < 0:
#                                 if len(buf) > 2_000_000:
#                                     buf = buf[-1_000_000:]
#                                 break

#                             # header end may be \r\n\r\n or \n\n
#                             hdr_end = buf.find(b"\r\n\r\n", start)
#                             sep_len = 4
#                             if hdr_end < 0:
#                                 hdr_end = buf.find(b"\n\n", start)
#                                 if hdr_end >= 0:
#                                     sep_len = 2
#                             if hdr_end < 0:
#                                 break  # need more data

#                             headers_block = buf[start+len(bnd):hdr_end]
#                             content_length = None
#                             for line in headers_block.splitlines():
#                                 ln = line.strip().lower()
#                                 if ln.startswith(b'content-length:'):
#                                     try:
#                                         content_length = int(line.split(b':', 1)[1].strip())
#                                     except Exception:
#                                         content_length = None
#                                     break

#                             payload_start = hdr_end + sep_len
#                             if content_length is not None:
#                                 end = payload_start + content_length
#                                 if len(buf) < end:
#                                     break
#                                 jpeg_bytes = buf[payload_start:end]
#                                 buf = buf[end:]
#                             else:
#                                 next_b = buf.find(bnd, payload_start)
#                                 if next_b < 0:
#                                     break
#                                 jpeg_bytes = buf[payload_start:next_b]
#                                 buf = buf[next_b:]

#                             # --- NEW BLOCK (paste this) ---
#                             arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                            
#                             # Prefer reduced decode when downscaling (saves CPU & time)
#                             flags = cv3.IMREAD_COLOR
#                             if self.width_scale <= 0.5 and self.height_scale <= 0.5:
#                                 flags = getattr(cv3, "IMREAD_REDUCED_COLOR_2", cv3.IMREAD_COLOR)
#                             elif self.width_scale <= 0.25 and self.height_scale <= 0.25:
#                                 flags = getattr(cv3, "IMREAD_REDUCED_COLOR_4", cv3.IMREAD_COLOR)
                            
#                             img = cv3.imdecode(arr, flags)
#                             if img is None:
#                                 continue
                            
#                             # If we couldn't reduce at decode time, do a normal resize
#                             if flags == cv3.IMREAD_COLOR and (self.width_scale != 1.0 or self.height_scale != 1.0):
#                                 h, w = img.shape[:2]
#                                 img = cv3.resize(
#                                     img,
#                                     (int(w * self.width_scale), int(h * self.height_scale)),
#                                     interpolation=cv3.INTER_AREA
#                                 )
                            
#                             # Drop if UI consumer is busy (prevents latency buildup)
#                             if not self.lock.acquire(blocking=False):
#                                 continue
#                             try:
#                                 self.frame = img
#                             finally:
#                                 self.lock.release()
                            
#             except Exception as e:
#                 if self.running:
#                     print("⚠ MJPEG reconnect due to:", e, "on", self.url)
#                 time.sleep(0.25)

#     def get_frame(self):
#         with self.lock:
#             return None if self.frame is None else self.frame.copy()


# # ========================= Main =========================

# if __name__ == "__main__":
#     # --- RTSP camera (keep behavior) ---
#     RTSP_URL = os.environ.get("RTSP_URL", "rtsp://192.168.1.99:1234")
#     cam_rtsp = RTSPCamera(RTSP_URL, width_scale=0.5, height_scale=0.5, latency_ms=0)
#     cam_rtsp.start()

#     # --- Two HTTP MJPEG streams served by Flask instances ---
#     host = "192.168.1.88"
#     base1 = f"http://{host}:9001"
#     base2 = f"http://{host}:9002"

#     cam_http1 = HTTPStreamCamera(base1, path="/video_feed", width_scale=0.5, height_scale=0.5)
#     cam_http2 = HTTPStreamCamera(base2, path="/video_feed", width_scale=0.5, height_scale=0.5)

#     # Only start if reachable (reduces error spam)
#     if is_port_open(host, 9001):
#         cam_http1.start()
#     else:
#         print(f"⚠ Port 9001 closed at {host}; skipping start for {base1}/video_feed")

#     if is_port_open(host, 9002):
#         cam_http2.start()
#     else:
#         print(f"⚠ Port 9002 closed at {host}; skipping start for {base2}/video_feed")

#     try:
#         while True:
#             fr_rtsp = cam_rtsp.get_frame()
#             if fr_rtsp is not None:
#                 cv3.imshow("RTSP Viewer - GStreamer", fr_rtsp)

#             fr1 = cam_http1.get_frame()
#             if fr1 is not None:
#                 cv3.imshow("HTTP MJPEG 192.168.1.88:9001/video_feed", fr1)

#             fr2 = cam_http2.get_frame()
#             if fr2 is not None:
#                 cv3.imshow("HTTP MJPEG 192.168.1.88:9002/video_feed", fr2)

#             if cv3.waitKey(1) & 0xFF == ord('q'):
#                 break
#             time.sleep(0.001)
#     finally:
#         cam_rtsp.stop()
#         cam_http1.stop()
#         cam_http2.stop()
#         try:
#             cv3.destroyAllWindows()
#         except Exception:
#             pass
