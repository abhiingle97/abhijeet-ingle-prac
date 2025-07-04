import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load foreground and background
foreground_image = cv2.imread("foreground.jpg")
background_image = cv2.imread("background.jpg")

if foreground_image is None or background_image is None:
    print("One or both images not found.")
    exit()

# Resize background to match foreground
background_image_resized = cv2.resize(background_image, (foreground_image.shape[1], foreground_image.shape[0]))

# Convert foreground to grayscale and apply threshold
foreground_gray = cv2.cvtColor(foreground_image, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(foreground_gray, 120, 255, cv2.THRESH_BINARY)

# Add alpha channel to foreground
foreground_with_alpha = cv2.cvtColor(foreground_image, cv2.COLOR_BGR2BGRA)
foreground_with_alpha[:, :, 3] = mask

# Display alpha-matted foreground
plt.imshow(cv2.cvtColor(foreground_with_alpha, cv2.COLOR_BGRA2RGBA))
plt.title("Foreground with Alpha Matte")
plt.axis("off")
plt.show()

# Normalize alpha channel to [0, 1]
alpha_channel = foreground_with_alpha[:, :, 3] / 255.0

# Extract RGB channels of foreground
foreground_rgb = foreground_with_alpha[:, :, :3]

# Blend images using alpha matte
blended = (alpha_channel[..., None] * foreground_rgb +
           (1 - alpha_channel[..., None]) * background_image_resized).astype(np.uint8)

# Display final composited image
plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
plt.title("Composited Image")
plt.axis("off")
plt.show()

# Save the result
cv2.imwrite("composited_output.jpg", blended)
print("✅ Output saved as composited_output.jpg")
