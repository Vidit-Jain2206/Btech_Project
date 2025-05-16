import cv2
import numpy as np
import os
from collections import defaultdict
from typing import Tuple, Dict

def detect_vehicles(image_filename: str) -> Tuple[int, Dict[str, int], np.ndarray]:
    # Load YOLOv3
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    # Load COCO class labels
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Vehicle-related classes
    vehicle_classes = ['car', 'motorbike', 'bus', 'truck', 'bicycle']

    # Assign distinct colors
    category_colors = {
        'car': (0, 255, 0),        # Green
        'motorbike': (255, 0, 0),  # Blue
        'bus': (0, 0, 255),        # Red
        'truck': (0, 255, 255),    # Yellow
        'bicycle': (255, 0, 255)   # Magenta
    }

    # Load image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, image_filename)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Could not load image: {image_path}")

    height, width = image.shape[:2]

    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)

    boxes, confidences, class_ids = [], [], []
    conf_threshold = 0.5
    nms_threshold = 0.4

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

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    vehicle_count = 0
    category_counts = defaultdict(int)

    for i in indices:
        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        confidence = confidences[i]

        color = category_colors.get(label, (255, 255, 255))
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label}: {confidence:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        vehicle_count += 1
        category_counts[label] += 1

    cv2.putText(image, f"Total Vehicles: {vehicle_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return vehicle_count, dict(category_counts), image



if __name__ == "__main__":
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vehicles_count2.jpg")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: Image file not found: {image_path}")
    total, per_category, output_image = detect_vehicles(image_path)

    print(f"\n✅ Total Vehicles Detected: {total}")
    print("📊 Category-wise Breakdown:")
    for category, count in per_category.items():
        print(f" - {category}: {count}")
    # Save the output image
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    image_path_arr = image_path.split("/")
    image_name = image_path_arr[-1]
    output_filename = os.path.join(output_dir, image_name)
    cv2.imwrite(output_filename, output_image)
    print(f"\n📸 Output image saved to: {output_filename}")
