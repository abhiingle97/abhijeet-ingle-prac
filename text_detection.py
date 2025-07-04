import cv2
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# Set path to tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
  # <-- change if needed

# Load image
img = cv2.imread("text_image.jpg")  # Replace with your image file

if img is None:
    print("Image not found!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Thresholding for better OCR
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Display thresholded image
plt.imshow(thresh, cmap='gray')
plt.title("Thresholded Image")
plt.axis('off')
plt.show()

# Perform OCR
text = pytesseract.image_to_string(thresh)
print("Extracted Text:\n", text)

# Draw bounding boxes around characters
h, w, _ = img.shape
boxes = pytesseract.image_to_boxes(img)

for b in boxes.splitlines():
    b = b.split()
    x, y, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(img, (x, h - y), (x2, h - y2), (0, 255, 0), 2)

# Show final image with boxes
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Detected Characters")
plt.axis('off')
plt.show()
