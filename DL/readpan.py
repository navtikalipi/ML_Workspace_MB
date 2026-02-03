import cv2
import pytesseract
import matplotlib.pyplot as plt
import re

# CHANGE THIS PATH (Windows only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Read PAN card image
img = cv2.imread("sample-pan-card.jpg")

if img is None:
    print("Image not found")
    exit()

# Convert BGR to RGB (for matplotlib)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Thresholding for better OCR
thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

# OCR
text = pytesseract.image_to_string(thresh)

print("RAW OCR TEXT:\n")
print(text)

# Extract PAN number using regex
pan_pattern = r"[A-Z]{5}[0-9]{4}[A-Z]"
pan = re.findall(pan_pattern, text)

print("\nExtracted PAN Number:")
if pan:
    print(pan[0])
else:
    print("PAN number not found")

# -------------------------
# Display images using plot
# -------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original PAN Card")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(thresh, cmap="gray")
plt.title("Thresholded (OCR Input)")
plt.axis("off")

plt.tight_layout()
plt.show()
