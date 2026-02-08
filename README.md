# Vision Clean – Smart De-Smoking & Object Detection System 👁️🚦

## Project Overview
Vision Clean is a computer vision–based system designed to improve visibility in smoky or hazy environments by removing smoke from visual input and performing object detection on the enhanced output. The project aims to improve detection accuracy in real-world scenarios such as traffic monitoring, surveillance, and industrial safety where smoke degrades image quality.

This system works on images, videos, and live camera feeds and shows a clear comparison between the original hazy input and the cleaned output with detected objects.

---

## Problem Statement
Smoke and haze significantly reduce the performance of computer vision systems by lowering contrast and visibility. Traditional object detection models struggle to identify objects accurately under such conditions.  
The challenge is to enhance visual quality first and then apply object detection to achieve better and more reliable results.

---

## Solution Approach
Vision Clean solves this problem in two stages:
1. **De-Smoking (Image Enhancement):**  
   Smoke is removed using image processing techniques to enhance contrast and visibility.
2. **Object Detection:**  
   YOLOv8 is applied on the de-smoked frames to detect objects more accurately.

The system processes the original and enhanced outputs side-by-side for better comparison.

---

## Features
- Smoke removal from images and videos
- Object detection using YOLOv8
- Real-time video and webcam support
- Side-by-side comparison of original vs cleaned output
- Works on hazy, smoky, and low-visibility scenes

---

## Technologies Used
- **Python**
- **OpenCV**
- **NumPy**
- **YOLOv8 (Ultralytics)**
- **Deep Learning & Computer Vision**

---

## Project Structure
