# YOLOv8 Multi-Object Tracker & Counter 🚀

> **Real-time object detection and tracking with state-of-the-art accuracy.**

---

## 📌 Project Overview

This project is a high-performance **Multi-Object Tracking (MOT)** system powered by **YOLOv8** and **DeepSORT**. It is designed to detect objects in video streams, assign unique IDs to them, and track their movement across frames. This solution is ideal for applications requiring robust traffic monitoring, crowd analytics, and surveillance automation.

## ✨ Features

- **Real-Time Detection**: Utilizes the speed and accuracy of YOLOv8.
- **Robust Tracking**: Maintains object identities across occlusions using DeepSORT/BoT-SORT algorithms.
- **Multi-Class Support**: capable of tracking cars, people, trucks, and more simultaneously.
- **Automated Counting**: Keeps a running tally of unique objects detected.
- **Video Export**: Automatically saves tracked footage for analysis.

## 🛠️ Tech Stack

- **Model**: [YOLOv8](https://github.com/ultralytics/ultralytics) (You Only Look Once)
- **Tracking**: DeepSORT / BoT-SORT
- **Language**: Python 3.12+
- **Libraries**: OpenCV, NumPy, PyTorch

## 📂 Project Structure

```
yolov8-tracker/
├── YOLOv4-DeepSort-Tensorflow-main/   # Core tracking implementation
│   ├── run_tracker_v8.py              # Main execution script (Modern)
│   ├── object_tracker.py              # Legacy tracking script
│   ├── save_model.py                  # Model conversion utilities
│   ├── data/                          # Input videos and assets
│   └── ...
├── yolov8n.pt                         # Pre-trained YOLOv8 weights (auto-downloaded)
├── output_v8.mp4                      # Generated output video
└── README.md                          # Project documentation
```

## ⚙️ How It Works

1.  **Frame Extraction**: The video input is processed frame-by-frame.
2.  **Detection**: YOLOv8 scans the frame and produces bounding boxes for objects.
3.  **Association**: The tracking algorithm matches new detections with existing objects based on:
    -   **IOU (Intersection Over Union)**: Spatial overlap.
    -   **Visual Feature Embeddings**: Appearance similarity.
4.  **Visualization**: Unique IDs and bounding boxes are drawn on the frame.

## 🎥 Demo Output

*(Replace with a screenshot or GIF of your output_v8.mp4)*

The system successfully identifies vehicles and maintains consistent IDs (e.g., `id: 1`) as they traverse the scene.

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/kiyansh13karan/yolov8-tracker.git
cd yolov8-tracker
```

### 2. Install Dependencies
```bash
pip install ultralytics opencv-python
```

### 3. Run the Tracker
To run the tracker on the default sample video:
```bash
cd YOLOv4-DeepSort-Tensorflow-main/YOLOv4-DeepSort-Tensorflow-main/YOLO-DeepSORT-Tensorflow
python run_tracker_v8.py --show
```

**Options:**
- `--video`: Path to specific input video.
- `--model`: Change model size (e.g., `yolov8s.pt`).

## 🔮 Future Improvements

- [ ] Implement speed estimation for vehicles.
- [ ] Add line-crossing analytics for directional counting.
- [ ] Deploy as a web API using Flask/FastAPI.
- [ ] Optimize for edge devices (Jetson Nano/Orin).

## 👏 Acknowledgements

- **Ultralytics** for the amazing YOLOv8 framework.
- **OpenCV** for powerful computer vision tools.

---

## 📬 Contact

**Karan Nayal**  
Computer Vision Enthusiast & Developer

- **GitHub**: [kiyansh13karan](https://github.com/kiyansh13karan)
- **LinkedIn**: [Karan Nayal](https://www.linkedin.com/in/karan-nayal-1313k/)
- **Email**: [karan13nayal@gmail.com](mailto:karan13nayal@gmail.com)
