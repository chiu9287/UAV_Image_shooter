from flask import Flask, render_template, Response, jsonify
import threading
import time
import os
import datetime
import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except Exception:
    rs = None

app = Flask(__name__)


class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.thread = None
        self.running = False
        self.latest_frame = None
        self.lock = threading.Lock()
        self.capturing = False
        self.capture_dir = None
        self.frame_counter = 0

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
                # no realsense available - produce a gray placeholder
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

            if self.capturing and self.capture_dir is not None:
                # save as JPEG
                filename = os.path.join(self.capture_dir, f"{self.frame_counter:06d}.jpg")
                try:
                    cv2.imwrite(filename, frame)
                    self.frame_counter += 1
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

    def start_capture(self, base_dir='captures'):
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(base_dir, exist_ok=True)
        folder = os.path.join(base_dir, ts)
        os.makedirs(folder, exist_ok=True)
        self.capture_dir = folder
        self.frame_counter = 0
        self.capturing = True
        return folder

    def stop_capture(self):
        self.capturing = False
        folder = self.capture_dir
        self.capture_dir = None
        return folder

    def shutdown(self):
        self.running = False
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
            # small wait to avoid busy-loop
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


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        camera.shutdown()
