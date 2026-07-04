# UAV Image Capture (Flask + RealSense)

Quick start:

1. Ensure the Intel RealSense SDK (librealsense) is installed on the system before installing `pyrealsense2`.
2. Create a virtualenv (optional) and install requirements:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

3. Run the server:

```bash
python app.py
```

4. Open a browser to `http://<device-ip>:5000` to view live feed and use the start/stop buttons. Captured images are saved under the `captures/` folder; each capture session creates a timestamped subfolder.

Notes:
- If your drone already runs Python 3.8.10, that version is compatible with Flask and OpenCV; keeping the system Python avoids environment problems.
- `pyrealsense2` needs matching librealsense binaries; on Linux install librealsense via your package manager or Intel instructions before `pip install pyrealsense2`.
