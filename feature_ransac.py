import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load images (grayscale)
img1 = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("image2.jpg", cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Error: One or both images not found.")
    exit()

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# Detect keypoints and descriptors
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

print("Number of keypoints in image 1:", len(keypoints1))
print("Number of keypoints in image 2:", len(keypoints2))

# Match using Brute-Force matcher with Hamming distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

print("Number of matches found:", len(matches))

# Extract location of good matches
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Compute Homography using RANSAC
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

print("Homography Matrix (H):\n", H)

# Draw top 10 matches
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display result
plt.figure(figsize=(15, 8))
plt.imshow(img_matches, cmap='gray')
plt.title("Top 10 Feature Matches using ORB + RANSAC")
plt.axis('off')
plt.show()
