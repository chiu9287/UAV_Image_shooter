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

        self.image_active = False
        self.recording_active = False
        self.image_dir = None
        self.video_dir = None
        self.video_writer = None
        self.video_path = None
        self.image_counter = 0

        if rs is not None:
            try:
                self.pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
                self.pipeline.start(cfg)
                print("RealSense camera connected successfully")
            except Exception as e:
                print(f"Failed to initialize RealSense: {e}")
                self.pipeline = None

        if self.pipeline is None:
            try:
                self.webcam = cv2.VideoCapture(0)
                if self.webcam.isOpened():
                    self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.webcam.set(cv2.CAP_PROP_FPS, self.fps)
                    print("Webcam available - using laptop camera")
                else:
                    self.webcam = None
                    print("No webcam detected - using synthetic test mode")
            except Exception as e:
                self.webcam = None
                print(f"Webcam initialization failed: {e}")

        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if self.pipeline is None:
                frame = None
                if self.webcam is not None and self.webcam.isOpened():
                    ok, frame = self.webcam.read()
                    if not ok or frame is None:
                        frame = None

                if frame is None:
                    # Generate synthetic test frame when no webcam or read failed
                    frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 50
                    color_idx = (self.test_frame_counter // self.fps) % 5
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                    color = colors[color_idx]
                    cv2.rectangle(frame, (100, 100), (100 + 200, 100 + 200), color, -1)
                    cv2.putText(frame, f'TEST FRAME #{self.test_frame_counter}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, 'No RealSense Camera Connected', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                with self.lock:
                    self.latest_frame = frame.copy()
                self.test_frame_counter += 1
                time.sleep(1.0 / max(1, self.fps))
                continue

            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            with self.lock:
                self.latest_frame = frame.copy()

            if self.image_active and self.image_dir is not None:
                filename = os.path.join(self.image_dir, f"{self.image_counter:06d}.jpg")
                try:
                    cv2.imwrite(filename, frame)
                    self.image_counter += 1
                except Exception:
                    pass

            if self.recording_active and self.video_writer is not None:
                try:
                    if not self.video_writer.isOpened():
                        print(f"Video writer is closed, stopping recording")
                        self.recording_active = False
                    else:
                        # Ensure frame is in BGR format for VideoWriter
                        ret = self.video_writer.write(frame)
                        if not ret:
                            print(f"Failed to write frame to video")
                except Exception as e:
                    print(f"Video write error: {e}")
                    self.recording_active = False

    def get_frame_bytes(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            ret, jpeg = cv2.imencode('.jpg', self.latest_frame)
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

        candidates = [
            (os.path.join(folder, 'recording.mp4'), cv2.VideoWriter_fourcc(*'mp4v')),
            (os.path.join(folder, 'recording.mp4'), cv2.VideoWriter_fourcc(*'avc1')),
            (os.path.join(folder, 'recording.mp4'), cv2.VideoWriter_fourcc(*'H264')),
            (os.path.join(folder, 'recording.mp4'), 0),
        ]

        self.video_writer = None
        self.video_path = None
        for video_path, fourcc in candidates:
            self.video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, (self.width, self.height))
            if self.video_writer.isOpened():
                self.video_path = video_path
                print(f"Video recording started at {video_path}")
                break
            else:
                print(f"Failed to open writer for {video_path}")

        if self.video_writer is None or not self.video_writer.isOpened():
            print(f"ERROR: VideoWriter failed to open for all candidates in {folder}")
        self.video_dir = folder
        self.recording_active = True
        return folder

    def stop_recording(self):
        self.recording_active = False
        if self.video_writer is not None:
            self.video_writer.release()
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


camera = RealSenseCamera()


@app.route('/')
def index():
    return render_template('index.html')


def gen_frames():
    while True:
        frame = camera.get_frame_bytes()
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_capture', methods=['POST'])
def start_capture():
    folder = camera.start_capture()
    return jsonify({'status': 'started', 'folder': folder})


@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    folder = camera.stop_capture()
    return jsonify({'status': 'stopped', 'folder': folder})


@app.route('/start_recording', methods=['POST'])
def start_recording():
    folder = camera.start_recording()
    return jsonify({'status': 'recording', 'folder': folder})


@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    folder = camera.stop_recording()
    return jsonify({'status': 'stopped', 'folder': folder})


@app.route('/api/status')
def api_status():
    return jsonify({
        'capture_active': camera.image_active,
        'recording_active': camera.recording_active,
        'fps': camera.fps,
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
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        camera.shutdown()
