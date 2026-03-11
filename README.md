# PY-Project: Hand Tracking & Gesture-Based Volume Control

A comprehensive Python project demonstrating real-time hand tracking, gesture detection, and gesture-based system volume control using computer vision techniques.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Module Descriptions](#module-descriptions)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project is a hands-on learning resource for computer vision enthusiasts. It combines **OpenCV** for video processing, **MediaPipe** for hand landmark detection, and **pycaw** for system audio control. The repository includes 8 progressive modules that build from basic video capture to a fully-featured gesture control system, plus an integrated final project.

Perfect for:
- Learning computer vision fundamentals
- Understanding hand pose estimation
- Building gesture-based interfaces
- Integrating ML models with system APIs

## Features

- Real-time Hand Detection: Identifies and tracks hand landmarks in live video
- Fingertip Recognition: Detects precise positions of all fingertip points
- Distance Measurement: Calculates Euclidean distance between finger points
- Dynamic Progress Visualization: Creates visual feedback based on gesture distance
- Gesture-Based Volume Control: Adjust system volume using finger pinch gestures
- FPS Monitoring: Real-time performance metrics display
- Progressive Learning Path: 8 modules from basics to advanced features

## Project Structure

```
PY-Project/
├── 1_basicvideocapturing.py              # Video capture fundamentals
├── 2_basichandrecognition.py             # Hand landmark detection
├── 3_Connecttipsoffigure.py              # Hand skeleton visualization
├── 4_addcircletotips.py                  # Fingertip visual markers
├── 5_showfps.py                          # Performance metrics
├── 6_Calculatingthedistancebetweenfigures.py  # Gesture distance calculation
├── 7_makingbarwithdistance.py            # Visual progress bar
├── 8_connectingwithsystemvolume.py       # Volume control implementation
├── Finalproject.py                       # Integrated final application
├── README.md                             # This file
└── requirements.txt                      # Dependency list
```

## Requirements

- **Python 3.7 or higher**
- **Webcam/Camera** (for video input)
- **Windows 10+, macOS 10.13+, or Linux** (Ubuntu 18.04+)

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | 4.5+ | Video capture & image processing |
| `mediapipe` | 0.8+ | Hand landmark detection |
| `pycaw` | Latest | Windows audio control |
| `comtypes` | Latest | Windows COM interface |

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/PY-Project.git
cd PY-Project
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install opencv-python mediapipe pycaw comtypes
```

### Step 3: Verify Setup

Test your installation by running a basic script:

```bash
python 1_basicvideocapturing.py
```

Press `q` to exit any script.

## Usage

### Run Individual Modules

Each script can be run independently to explore specific features:

```bash
# Basic video capture
python 1_basicvideocapturing.py

# Hand recognition
python 2_basichandrecognition.py

# Hand structure visualization
python 3_Connecttipsoffigure.py

# Fingertip markers
python 4_addcircletotips.py

# FPS display
python 5_showfps.py

# Distance calculation
python 6_Calculatingthedistancebetweenfigures.py

# Volume bar visualization
python 7_makingbarwithdistance.py

# System volume control
python 8_connectingwithsystemvolume.py
```

### Run the Final Integrated Project

```bash
python Finalproject.py
```

**Gesture Controls:**
- **Pinch Gesture**: Bring your thumb and index finger close together to adjust volume
- **Move Hand Up/Down**: Control progress bar or volume level
- **Open Palm**: Reset/neutral gesture

## Module Descriptions

| Module | Purpose | Key Learnings |
|--------|---------|---------------|
| **1_basicvideocapturing.py** | Foundational OpenCV video capture | Video streams, frame processing, display loops |
| **2_basichandrecognition.py** | Hand detection using MediaPipe | Pose estimation, landmark extraction |
| **3_Connecttipsoffigure.py** | Draw hand skeleton overlay | Geometric visualization, coordinate mapping |
| **4_addcircletotips.py** | Visual fingertip detection | Circle drawing, point detection |
| **5_showfps.py** | Real-time performance metrics | Timing, FPS calculation |
| **6_Calculatingthedistancebetweenfigures.py** | Euclidean distance computation | Mathematical operations, gesture quantification |
| **7_makingbarwithdistance.py** | Dynamic progress bar | Mapping distance to UI elements |
| **8_connectingwithsystemvolume.py** | OS volume control | System API integration, pycaw library |
| **Finalproject.py** | Complete integrated application | Full pipeline: detection → calculation → control |

## How It Works

### Hand Detection Pipeline

```
Webcam Input
    ↓
OpenCV Frame Capture
    ↓
MediaPipe Hand Detection
    ↓
Landmark Extraction (21 hand points)
    ↓
Distance Calculation & Gesture Recognition
    ↓
System Volume Control / Visual Feedback
    ↓
Display Output
```

### Key Components

**MediaPipe Hand Landmarks:**
- 21 hand landmarks per detected hand
- Includes: wrist, palm, fingers, and fingertips
- Real-time detection with <100ms latency

**Gesture Recognition:**
- Pinch detection via thumb-index distance
- Distance-to-volume mapping
- Normalized coordinate system

**System Integration:**
- Windows audio API via `pycaw`
- Supports dynamic volume adjustment
- Real-time feedback visualization

## Gesture Examples

### Volume Control Gesture
```
Distance between thumb & index finger:
│ Closed (0-30px)  │ Partial (30-150px) │ Open (150-300px) │
│   Volume: 0%     │  Volume: 50%       │  Volume: 100%    │
```

## Troubleshooting

### Camera Not Detected
```python
# Check available cameras in your system
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} available")
    cap.release()
```

### Low Detection Accuracy
- Ensure adequate lighting
- Keep hands within frame center
- Avoid extreme hand rotations
- Increase camera resolution if possible

### Volume Control Not Working (Windows)
- Ensure speaker/audio device is available
- Run script with administrator privileges
- Check system volume is not muted
- Verify `pycaw` installation: `pip install --upgrade pycaw`

### macOS / Linux Users
Volume control features require platform-specific audio APIs. The gesture detection modules (1-7) work on all platforms, but `Finalproject.py` requires modification for macOS/Linux.

**macOS Alternative:**
```python
import osascript
osascript.execute(f'set volume output volume {volume_percent}')
```

**Linux Alternative:**
```bash
amixer set Master {percentage}%
```

## Performance Tips

- Run at 1280×720 resolution for optimal balance
- Reduce detection confidence threshold if tracking is unstable
- Use GPU acceleration if available (CUDA/OpenCL)
- Close unnecessary background applications

## Customization Ideas

- Add multi-hand gesture recognition
- Implement hand pose classification (rock/paper/scissors)
- Build custom gesture trainer
- Integrate with other system controls (brightness, application switching)
- Add machine learning-based gesture classifier
- Export hand tracking data for analysis

## Code Examples

### Basic Hand Detection
```python
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    cv2.imshow('Hand Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Distance Calculation
```python
import math

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Usage
thumb_tip = (100, 150)
index_tip = (120, 180)
distance = calculate_distance(thumb_tip, index_tip)
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

### Guidelines
- Write clean, documented code
- Add comments for complex logic
- Test on multiple devices/OS
- Update README for new features

## License

This project is licensed under the MIT License – see the LICENSE file for details.

You are free to:
- Use commercially
- Modify the code
- Distribute copies
- Use privately

With the condition:
- Include the license and copyright notice

## Author

Created as an educational resource for computer vision learning.

## Acknowledgments

- OpenCV team for excellent computer vision library
- Google MediaPipe for state-of-the-art hand detection
- pycaw contributors for Windows audio control

## Support and Contact

- Issues: Open an issue on GitHub for bugs or questions
- Discussions: Use GitHub Discussions for feature requests
- Email: your-garvitjain131@gmail.com

## Resources

- MediaPipe Documentation: https://google.github.io/mediapipe/
- OpenCV Tutorials: https://docs.opencv.org/master/d9/df8/tutorial_root.html
- Hand Pose Estimation Paper: https://arxiv.org/abs/2006.10214
- pycaw GitHub: https://github.com/AndreMiras/pycaw

---

Last Updated: March 2026 | Python Version: 3.7+ | Status: Active Development
