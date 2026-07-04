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
    def __init__(self, width=640, height=480, fps=60):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.thread = None
        self.running = False
        self.latest_frame = None
        self.lock = threading.Lock()

        self.image_active = False
        self.recording_active = False
        self.image_dir = None
        self.video_dir = None
        self.video_writer = None
        self.video_path = None
        self.image_counter = 0

        if rs is not None:
            self.pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self.pipeline.start(cfg)

        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if self.pipeline is None:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.putText(frame, 'No RealSense (pyrealsense2 missing)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                with self.lock:
                    self.latest_frame = frame
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
                    self.video_writer.write(frame)
                except Exception:
                    pass

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
        video_path = os.path.join(folder, 'recording.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, (self.width, self.height))
        self.video_dir = folder
        self.video_path = video_path
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


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        camera.shutdown()
