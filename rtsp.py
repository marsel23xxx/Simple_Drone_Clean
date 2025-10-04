def init_cameras(self):
        """Initialize cameras with error handling - but don't start them"""
        # Main RTSP camera
        rtsp_url = os.environ.get("RTSP_URL", "rtsp://192.168.1.99:1234")
        try:
            self.cameras["rtsp"] = RTSPCamera(rtsp_url, width_scale=0.8, height_scale=0.8)
            self.log_message(f"✓ RTSP camera created: {rtsp_url} (disabled)")
        except Exception as e:
            self.log_message(f"✗ RTSP camera failed: {e}")
        
        # RTSP cameras - bottom and top
        host = os.environ.get("CAMERA_HOST", "192.168.1.88")
        
        # Camera 1 - Bottom
        try:
            bottom_url = f"rtsp://{host}:8555/bottom"
            self.cameras["http1"] = RTSPCamera(bottom_url, width_scale=0.6, height_scale=0.6, latency_ms=0)
            self.log_message(f"✓ Camera 1 (Bottom) created: {bottom_url} (disabled)")
        except Exception as e:
            self.log_message(f"✗ Camera 1 failed: {e}")
        
        # Camera 2 - Top
        try:
            top_url = f"rtsp://{host}:8554/top"
            self.cameras["http2"] = RTSPCamera(top_url, width_scale=0.6, height_scale=0.6, latency_ms=0)
            self.log_message(f"✓ Camera 2 (Top) created: {top_url} (disabled)")
        except Exception as e:
            self.log_message(f"✗ Camera 2 failed: {e}")