import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8s.pt")   

# Run detection
results = model("catss.jpg")

cat_count = 0
TARGET_CLASS = "cat"

# Read image (for drawing)
img = cv2.imread("catss.jpg")

for r in results:
    for box in r.boxes:
        class_id = int(box.cls[0])
        class_name = r.names[class_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        #lower confidence threshold
        if class_name == TARGET_CLASS and confidence > 0.25:
            cat_count += 1

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label with confidence
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(
                img, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

            # Console output
            print(
                "Class:", class_name,
                "| Confidence:", round(confidence, 2),
                "| Coordinates:", (x1, y1, x2, y2)
            )

# Convert to RGB for plotting
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Show image
plt.figure(figsize=(8, 6))
plt.imshow(img_rgb)
plt.title(f"Total Cats Detected: {cat_count}")
plt.axis("off")
plt.show()

print("\n✅ Final Cat Count:", cat_count)
