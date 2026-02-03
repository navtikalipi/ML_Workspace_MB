import cv2
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("cat.jpg")

if img is None:
    print("Image not found")
    exit()

# Convert BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 🔴 CHANGED: increased blur kernel (5,5) → (7,7)
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# 🔴 CHANGED: lower Canny thresholds for low-contrast image
edges = cv2.Canny(blur, 30, 100)

# Display images
plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(blur, cmap="gray")
plt.title("Blur")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(edges, cmap="gray")
plt.title("Edges")
plt.axis("off")

plt.show()
