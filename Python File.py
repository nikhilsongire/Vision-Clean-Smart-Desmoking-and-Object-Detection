import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO  # Importing YOLOv8

# Load YOLOv8 model
try:
    model = YOLO("yolov8n.pt")  # Load YOLOv8 nano model
except Exception as e:
    messagebox.showerror("Error", f"Failed to load YOLOv8 model: {e}")

# Smoke removal function
def remove_smog(image):
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel = cv2.erode(min_channel, kernel)

    top_brightest_fraction = 0.1
    num_brightest_pixels = int(top_brightest_fraction * dark_channel.size)
    indices = np.argsort(dark_channel.ravel())[-num_brightest_pixels:]
    atmospheric_light = np.mean(
        [image[i // dark_channel.shape[1], i % dark_channel.shape[1]] for i in indices], axis=0
    )

    omega = 0.95
    transmission = 1 - omega * (dark_channel / atmospheric_light.max())

    transmission_blurred = cv2.GaussianBlur(transmission, (15, 15), 0)

    t_min = 0.1
    dehazed_image = (image - atmospheric_light) / np.clip(transmission_blurred[..., None], t_min, 1) + atmospheric_light
    dehazed_image = np.clip(dehazed_image, 0, 255).astype(np.uint8)

    return dehazed_image

# Object detection function using YOLOv8
def detect_objects(img):
    try:
        results = model(img)
        detections = results[0]

        bounding_boxes = []
        confidences = []
        class_names = []

        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_name = model.names[int(box.cls[0])]

            if confidence > 0.5:
                bounding_boxes.append((x1, y1, x2, y2))
                confidences.append(confidence)
                class_names.append(class_name)

        return bounding_boxes, confidences, class_names
    except Exception as e:
        messagebox.showerror("Error", f"YOLOv8 detection failed: {e}")
        return [], [], []

# Tkinter GUI Setup
root = tk.Tk()
root.title("Desmoking and Object Detection")
root.geometry("900x700")

# Styling
style = ttk.Style()
style.configure("TFrame", background="#f0f0f0")
style.configure("TLabel", background="#f0f0f0", font=("Arial", 12))
style.configure("TButton", font=("Arial", 10, "bold"), foreground="white", background="#007acc")

# Main frames
header_frame = ttk.Frame(root)
header_frame.pack(fill="x", pady=10)

content_frame = ttk.Frame(root)
content_frame.pack(fill="both", expand=True, padx=10, pady=10)

footer_frame = ttk.Frame(root)
footer_frame.pack(fill="x", pady=10)

# Header
header_label = ttk.Label(header_frame, text="Desmoking and Object Detection System", font=("Arial", 16, "bold"))
header_label.pack(pady=5)

# Content layout
left_frame = ttk.Frame(content_frame, width=300, height=500)
left_frame.pack(side="left", fill="y", padx=10, pady=10)

right_frame = ttk.Frame(content_frame, width=600, height=500)
right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

# Left Frame Buttons
def create_button(text, command):
    return ttk.Button(left_frame, text=text, command=command)

tk.Label(left_frame, text="Actions:", font=("Arial", 14, "bold"), background="#f0f0f0").pack(pady=10)
create_button("Load Image", lambda: load_image()).pack(fill="x", pady=5)
create_button("Remove Smoke & Detect", lambda: remove_smoke_and_detect()).pack(fill="x", pady=5)
create_button("Load Video", lambda: load_video()).pack(fill="x", pady=5)
create_button("Live Feed", lambda: process_video()).pack(fill="x", pady=5)

# Image Display Area
video_label = ttk.Label(right_frame)
video_label.pack(expand=True)

# Functions
def load_image():
    try:
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")],
        )
        if image_path:
            img = Image.open(image_path)
            img.thumbnail((400, 400))
            img_tk = ImageTk.PhotoImage(img)
            video_label.config(image=img_tk)
            video_label.image = img_tk
            root.image_path = image_path
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {e}")


def remove_smoke_and_detect():
    try:
        if hasattr(root, "image_path"):
            img_cv = cv2.imread(root.image_path)
            dehazed_image = remove_smog(img_cv)
            bounding_boxes, _, class_names = detect_objects(dehazed_image)

            for (x1, y1, x2, y2), label in zip(bounding_boxes, class_names):
                cv2.rectangle(dehazed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(dehazed_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            corrected_image_path = "corrected_image.jpg"
            cv2.imwrite(corrected_image_path, dehazed_image)
            img_pil = Image.open(corrected_image_path)
            img_pil.thumbnail((400, 400))
            img_tk = ImageTk.PhotoImage(img_pil)
            video_label.config(image=img_tk)
            video_label.image = img_tk
        else:
            messagebox.showwarning("No Image Loaded", "Please load an image first.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process image: {e}")

# Additional functions for video and live feed (not shown to save space)

root.mainloop()
