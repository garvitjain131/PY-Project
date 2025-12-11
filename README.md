PY-Project

Hand Tracking, Gesture Detection & System Volume Control using Python, OpenCV, and MediaPipe

This project contains multiple Python scripts that demonstrate different computer-vision techniques such as video capturing, hand recognition, fingertip detection, distance measurement, progress bar creation, and gesture-based system volume control.

Project Files

1_basicvideocapturing.py – Captures video from the webcam using OpenCV

2_basichandrecognition.py – Detects and tracks hands using MediaPipe

3_Connecttipsoffigure.py – Connects fingertip points to visualize hand structure

4_addcircletotips.py – Draws circles on each fingertip landmark

5_showfps.py – Shows the FPS (Frames Per Second) on the video stream

6_Calculatingthedistancebetweenfigures.py – Calculates the distance between two finger points

7_makingbarwithdistance.py – Creates a progress/volume bar based on finger distance

8_connectingwithsystemvolume.py – Controls system volume using hand gestures

Finalproject.py – Fully integrated version combining all features

Features

Real-time hand tracking

Fingertip detection

Gesture-based distance measurement

Dynamic progress bar

Volume control using finger distance

Step-by-step scripts for learning computer vision

Requirements

Install all necessary libraries:

pip install opencv-python mediapipe pycaw comtypes

How to Run

Run any script using:

python filename.py


Example:

python Finalproject.py


Make sure your webcam is connected.

Project Structure
PY-Project/
│── 1_basicvideocapturing.py
│── 2_basichandrecognition.py
│── 3_Connecttipsoffigure.py
│── 4_addcircletotips.py
│── 5_showfps.py
│── 6_Calculatingthedistancebetweenfigures.py
│── 7_makingbarwithdistance.py
│── 8_connectingwithsystemvolume.py
│── Finalproject.py
│── README.md

Contributing

Feel free to fork this repository and add new gesture-based features.

License

This project uses the MIT License.
