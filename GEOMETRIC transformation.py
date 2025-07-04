import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load local image (replace 'test.jpg' with your image file)
image_path = "test.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Image not found. Please check the path or file name.")
    exit()

# Convert BGR to RGB for matplotlib display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 1. Original Image
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()

# 2. Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

# 3. Gaussian Blur + Canny Edges
blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
edges = cv2.Canny(blurred, 100, 200)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')
plt.show()

# 4. Draw Rectangle
img_copy = image.copy()
cv2.rectangle(img_copy, (30, 30), (130, 130), (0, 255, 0), 3)
plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
plt.title("Rectangle on Image")
plt.axis('off')
plt.show()

# 5. Resize
resized = cv2.resize(image, (300, 300))
plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
plt.title("Resized Image")
plt.axis('off')
plt.show()

# 6. Thresholding
_, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
plt.imshow(thresholded, cmap='gray')
plt.title("Thresholded Image")
plt.axis('off')
plt.show()

# 7. Rotation using PIL
img_pil = Image.open(image_path)
rotated = img_pil.rotate(90)
plt.imshow(rotated)
plt.title("Rotated Image (90°)")
plt.axis('off')
plt.show()
