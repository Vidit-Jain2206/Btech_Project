import cv2
import numpy as np
import os
from collections import defaultdict

# Load YOLOv3
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Vehicle-related classes to detect
vehicle_classes = ['car', 'motorbike', 'bus', 'truck', 'bicycle']

# Assign distinct colors to each vehicle category
category_colors = {
    'car': (0, 255, 0),        # Green
    'motorbike': (255, 0, 0),  # Blue
    'bus': (0, 0, 255),        # Red
    'truck': (0, 255, 255),    # Yellow
    'bicycle': (255, 0, 255)   # Magenta
}

# Load the image
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'vehicles_count3.jpg')
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found or unable to load.")
    exit()

height, width = image.shape[:2]

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Get YOLO output layers
layer_names = net.getUnconnectedOutLayersNames()
outputs = net.forward(layer_names)

# Store detection data
boxes = []
confidences = []
class_ids = []

# Detection thresholds
conf_threshold = 0.5
nms_threshold = 0.4

# Parse outputs
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        label = classes[class_id]

        if confidence > conf_threshold and label in vehicle_classes:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Count total and per category
vehicle_count = 0
category_counts = defaultdict(int)

# Draw detections
for i in indices:
    i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
    box = boxes[i]
    x, y, w, h = box
    label = classes[class_ids[i]]
    confidence = confidences[i]
    
    vehicle_count += 1
    category_counts[label] += 1

    color = category_colors.get(label, (255, 255, 255))  # Default: white

    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    text = f"{label}: {confidence:.2f}"
    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Print counts
print(f"\n✅ Total Vehicles Detected: {vehicle_count}")
print("📊 Category-wise Breakdown:")
for vehicle_type, count in category_counts.items():
    print(f" - {vehicle_type}: {count}")

# Show total count on image
cv2.putText(image, f"Total Vehicles: {vehicle_count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Detected Vehicles", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
