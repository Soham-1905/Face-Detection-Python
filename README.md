Hello!

Overview:
This project demonstrates a face detection system using Python and OpenCV. The program can detect faces and facial features such as eyes, nose, and mouth in real-time through webcam input.

Features:
Real-time face detection using webcam
Detection of facial features: eyes, nose, and mouth
Visual indication of detected features with bounding boxes and labels

Functions Defined:

draw_boundary Function
This function detects features in an image and draws a bounding box around them.
Parameters:
img: The input image
classifier: The Haar Cascade classifier for detection
scaleFactor: The scale factor for detection
minNeighbor: The minimum number of neighbors for detection
color: The color of the bounding box
text: The label text for the bounding box

detect Function
This function handles the overall detection process for faces and facial features.
Parameters:
img: The input image
faceCascade: The Haar Cascade classifier for face detection
eyesCascade: The Haar Cascade classifier for eyes detection
noseCascade: The Haar Cascade classifier for nose detection
mouthCascade: The Haar Cascade classifier for mouth detection

