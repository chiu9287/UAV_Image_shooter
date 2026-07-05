from flask import Flask, render_template, Response, jsonify, send_from_directory
import threading
import time
import os
import datetime
import cv2
import numpy as np
from urllib.parse import quote

try:
    import pyrealsense2 as rs
except Exception:
    rs = None

app = Flask(__name__)


class RealSenseCamera:
    def __init__(self, width=1920, height=1080, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.webcam = None
        self.thread = None
        self.running = False
        self.latest_frame = None
        self.lock = threading.Lock()
        self.test_frame_counter = 0  # for generating test frames
        self.last_reconnect_attempt = 0.0
        self.reconnect_cooldown = 2.0
        self.preview_width = 480
        self.preview_height = 270
        self.preview_fps = 20
        self.jpeg_quality = 55
        self.preview_frame = None
        self.preview_bytes = None
        self.preview_last_sent_at = 0.0

        self.image_active = False
        self.recording_active = False
        self.image_dir = None
        self.video_dir = None
        self.video_writer = None
        self.video_path = None
        self.image_counter = 0
        self.recording_width = self.width
        self.recording_height = self.height
        self.recording_fps = self.fps
        self._recording_fallback_index = 0

        self._initialize_realsense_pipeline()

        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _initialize_realsense_pipeline(self):
        if rs is None:
            print("pyrealsense2 is not available")
            self.pipeline = None
            return False

        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None

        try:
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            pipeline.start(cfg)
            self.pipeline = pipeline
            self.last_reconnect_attempt = 0.0
            print("RealSense camera connected successfully")
            return True
        except Exception as e:
            self.pipeline = None
            print(f"RealSense initialization failed: {e}")
            return False

    def _ensure_realsense_pipeline(self):
        now = time.time()
        if self.pipeline is not None:
            return True
        if now - self.last_reconnect_attempt < self.reconnect_cooldown:
            return False
        self.last_reconnect_attempt = now
        return self._initialize_realsense_pipeline()

    def _normalize_frame(self, frame):
        if frame is None:
            return None
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        return frame

    def _start_video_writer(self, width=None, height=None, fps=None):
        width = width or self.recording_width or self.width
        height = height or self.recording_height or self.height
        fps = fps or self.recording_fps or self.fps

        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None

        if self.video_dir is None:
            return False

        # First, try GStreamer pipeline writer (preferred on Jetson)
        try:
            video_path = os.path.join(self.video_dir, 'recording.mp4')
            # GStreamer pipeline using x264enc (software) or hardware enc if available
            gst_pipeline = (
                f"appsrc ! videoconvert ! "
                f"x264enc tune=zerolatency speed-preset=superfast ! mp4mux ! filesink location={video_path} sync=false"
            )
            writer = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, fps, (width, height), True)
            if writer.isOpened():
                with self.lock:
                    self.video_writer = writer
                self.video_path = video_path
                self.recording_width = width
                self.recording_height = height
                self.recording_fps = fps
                print(f"GStreamer Video writer ready: {width}x{height}@{fps} fps -> {video_path}")
                return True
            else:
                print("GStreamer writer failed to open, falling back to native writers")
        except Exception as e:
            print(f"GStreamer writer attempt failed: {e}")

        # Fallback: try common fourcc writers (AVI containers more stable)
        candidates = [
            (os.path.join(self.video_dir, 'recording.avi'), cv2.VideoWriter_fourcc(*'MJPG')),
            (os.path.join(self.video_dir, 'recording.avi'), cv2.VideoWriter_fourcc(*'XVID')),
            (os.path.join(self.video_dir, 'recording.avi'), 0),
        ]

        for video_path, fourcc in candidates:
            try:
                writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                if writer.isOpened():
                    with self.lock:
                        self.video_writer = writer
                    self.video_path = video_path
                    self.recording_width = width
                    self.recording_height = height
                    self.recording_fps = fps
                    print(f"Video writer ready: {width}x{height}@{fps} fps -> {video_path}")
                    return True
                else:
                    print(f"Failed to open writer for {video_path} with {width}x{height}@{fps}")
            except Exception as e:
                print(f"Error opening writer {video_path}: {e}")

        return False
    def _fallback_recording_profile(self):
        profiles = [
            (self.width, self.height, self.fps),
            (1280, 720, 20),
            (960, 540, 15),
            (640, 480, 10),
        ]
        idx = self._recording_fallback_index % len(profiles)
        self._recording_fallback_index += 1
        return profiles[idx]

    def _handle_realsense_frame_error(self, error):
        error_text = str(error).lower()
        if self.pipeline is None:
            return False

        if (
            "frame didn't arrive" in error_text
            or "wait_for_frames" in error_text
            or "timed out" in error_text
            or "disconnected" in error_text
            or "not available" in error_text
        ):
            print(f"RealSense frame timeout or read error: {error}")
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
            self._initialize_realsense_pipeline()
            return True

        return False

    def _prepare_preview_frame(self, frame):
        if frame is None:
            self.preview_frame = None
            self.preview_bytes = None
            return

        normalized = self._normalize_frame(frame)
        if normalized is None:
            self.preview_frame = None
            self.preview_bytes = None
            return

        if normalized.shape[1] != self.preview_width or normalized.shape[0] != self.preview_height:
            resized = cv2.resize(normalized, (self.preview_width, self.preview_height), interpolation=cv2.INTER_AREA)
        else:
            resized = normalized

        self.preview_frame = resized
        if self.preview_frame.dtype != np.uint8:
            self.preview_frame = self.preview_frame.astype(np.uint8)

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        _, jpg = cv2.imencode('.jpg', self.preview_frame, encode_params)
        self.preview_bytes = jpg.tobytes() if jpg is not None else None

    def _get_idle_frame(self):
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def _update(self):
        while self.running:
            if self.pipeline is None:
                if not self._ensure_realsense_pipeline():
                    frame = self._get_idle_frame()
                    frame = self._normalize_frame(frame)
                    with self.lock:
                        self.latest_frame = frame.copy()
                    self.test_frame_counter += 1
                    time.sleep(0.2)
                    continue

            try:
                frames = self.pipeline.wait_for_frames()
            except Exception as exc:
                if self._handle_realsense_frame_error(exc):
                    continue
                print(f"Unexpected RealSense read error: {exc}")
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            frame = self._normalize_frame(frame)

            with self.lock:
                self.latest_frame = frame.copy()
                self._prepare_preview_frame(frame)

            if self.image_active and self.image_dir is not None:
                filename = os.path.join(self.image_dir, f"{self.image_counter:06d}.jpg")
                try:
                    cv2.imwrite(filename, frame)
                    self.image_counter += 1
                except Exception:
                    pass

            if self.recording_active and self.video_writer is not None:
                try:
                    with self.lock:
                        writer = self.video_writer
                        rw = self.recording_width
                        rh = self.recording_height
                    if writer is None or not writer.isOpened():
                        print("Video writer is closed or None, stopping recording")
                        self.recording_active = False
                    else:
                        # Ensure frame matches writer size
                        if (frame.shape[1], frame.shape[0]) != (rw, rh):
                            write_frame = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)
                        else:
                            write_frame = frame
                        try:
                            # VideoWriter.write() returns None in OpenCV Python bindings; rely on exceptions and isOpened
                            writer.write(write_frame)
                        except Exception as e:
                            print(f"Failed to write frame to video (exception): {e}")
                            width, height, fps = self._fallback_recording_profile()
                            print(f"Trying fallback recording profile: {width}x{height}@{fps} fps")
                            if self._start_video_writer(width, height, fps):
                                with self.lock:
                                    self.video_writer.write(cv2.resize(frame, (self.recording_width, self.recording_height), interpolation=cv2.INTER_AREA))
                            else:
                                print("Recording could not be recovered")
                                self.recording_active = False
                except Exception as e:
                    print(f"Video write error: {e}")
                    self.recording_active = False

    def get_frame_bytes(self):
        if not hasattr(self, 'preview_bytes'):
            self.preview_bytes = None
        if not hasattr(self, 'preview_frame'):
            self.preview_frame = None
        if not hasattr(self, 'preview_last_sent_at'):
            self.preview_last_sent_at = 0.0
        if not hasattr(self, 'preview_fps'):
            self.preview_fps = 10
        with self.lock:
            if self.preview_bytes is not None and self.preview_frame is not None:
                now = time.time()
                if now - self.preview_last_sent_at >= 1.0 / max(1, self.preview_fps):
                    self.preview_last_sent_at = now
                    return self.preview_bytes
                return None

            frame = self.latest_frame
            if frame is None:
                frame = self._get_idle_frame()
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                return None
            return jpeg.tobytes()

    def start_capture(self, base_dir='captures/image'):
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(base_dir, exist_ok=True)
        folder = os.path.join(base_dir, ts)
        os.makedirs(folder, exist_ok=True)
        self.image_dir = folder
        self.image_counter = 0
        self.image_active = True
        return folder

    def stop_capture(self):
        self.image_active = False
        folder = self.image_dir
        self.image_dir = None
        return folder

    def start_recording(self, base_dir='captures/video'):
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(base_dir, exist_ok=True)
        folder = os.path.join(base_dir, ts)
        os.makedirs(folder, exist_ok=True)

        self.video_writer = None
        self.video_path = None
        self.video_dir = folder
        self._recording_fallback_index = 0
        self.recording_active = True

        profiles = [
            (self.width, self.height, self.fps),
            (1280, 720, 20),
            (960, 540, 15),
            (640, 480, 10),
        ]
        for width, height, fps in profiles:
            if self._start_video_writer(width, height, fps):
                print(f"Video recording started at {self.video_path} ({width}x{height}@{fps})")
                break

        if self.video_writer is None or not self.video_writer.isOpened():
            print(f"ERROR: VideoWriter failed to open for all candidates in {folder}")
            self.recording_active = False
        return folder
    def stop_recording(self):
        self.recording_active = False
        with self.lock:
            if self.video_writer is not None:
                try:
                    self.video_writer.release()
                except Exception:
                    pass
                self.video_writer = None
        folder = self.video_dir
        self.video_dir = None
        self.video_path = None
        return folder

    def shutdown(self):
        self.running = False
        self.image_active = False
        self.recording_active = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        if self.webcam is not None:
            try:
                self.webcam.release()
            except Exception:
                pass
            self.webcam = None


camera = None


def get_camera():
    global camera
    if camera is None:
        camera = RealSenseCamera()
    return camera


@app.route('/')
def index():
    return render_template('index.html')


def gen_frames():
    cam = get_camera()
    while True:
        frame = cam.get_frame_bytes()
        if frame is None:
            time.sleep(0.03)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_capture', methods=['POST'])
def start_capture():
    folder = get_camera().start_capture()
    return jsonify({'status': 'started', 'folder': folder})


@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    folder = get_camera().stop_capture()
    return jsonify({'status': 'stopped', 'folder': folder})


@app.route('/start_recording', methods=['POST'])
def start_recording():
    folder = get_camera().start_recording()
    return jsonify({'status': 'recording', 'folder': folder})


@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    folder = get_camera().stop_recording()
    return jsonify({'status': 'stopped', 'folder': folder})


@app.route('/api/status')
def api_status():
    cam = get_camera()
    return jsonify({
        'capture_active': cam.image_active,
        'recording_active': cam.recording_active,
        'fps': cam.fps,
    })


@app.route('/api/folders')
def api_folders():
    image_root = os.path.join(app.root_path, 'captures', 'image')
    video_root = os.path.join(app.root_path, 'captures', 'video')
    image_folders = []
    video_folders = []

    if os.path.isdir(image_root):
        image_folders = [name for name in sorted(os.listdir(image_root)) if os.path.isdir(os.path.join(image_root, name))]
    if os.path.isdir(video_root):
        video_folders = [name for name in sorted(os.listdir(video_root)) if os.path.isdir(os.path.join(video_root, name))]

    return jsonify({'image_folders': image_folders, 'video_folders': video_folders})


@app.route('/api/images/<path:folder_name>')
def api_images(folder_name):
    safe_folder = os.path.basename(folder_name)
    folder = os.path.join(app.root_path, 'captures', 'image', safe_folder)
    if not os.path.isdir(folder):
        return jsonify({'files': []}), 404

    files = [f for f in sorted(os.listdir(folder)) if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_urls = [f"/api/image_file/{quote(safe_folder)}/{quote(f)}" for f in files]
    return jsonify({'folder': safe_folder, 'files': files, 'image_urls': image_urls})


@app.route('/api/image_file/<path:folder_name>/<path:filename>')
def api_image_file(folder_name, filename):
    safe_folder = os.path.basename(folder_name)
    safe_filename = os.path.basename(filename)
    folder = os.path.join(app.root_path, 'captures', 'image', safe_folder)
    return send_from_directory(folder, safe_filename)


@app.route('/api/videos/<path:folder_name>')
def api_videos(folder_name):
    safe_folder = os.path.basename(folder_name)
    folder = os.path.join(app.root_path, 'captures', 'video', safe_folder)
    if not os.path.isdir(folder):
        return jsonify({'files': []}), 404

    files = [f for f in sorted(os.listdir(folder)) if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    video_urls = [f"/api/video_file/{quote(safe_folder)}/{quote(f)}" for f in files]
    return jsonify({'folder': safe_folder, 'files': files, 'video_urls': video_urls})


@app.route('/api/video_file/<path:folder_name>/<path:filename>')
def api_video_file(folder_name, filename):
    safe_folder = os.path.basename(folder_name)
    safe_filename = os.path.basename(filename)
    folder = os.path.join(app.root_path, 'captures', 'video', safe_folder)
    return send_from_directory(folder, safe_filename)


if __name__ == '__main__':
    cam = get_camera()
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        if cam is not None:
            cam.shutdown()
