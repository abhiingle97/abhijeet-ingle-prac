import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load 3 local images
img1 = cv2.imread("image1.jpg")
img2 = cv2.imread("image2.jpg")
img3 = cv2.imread("image3.jpg")

if img1 is None or img2 is None or img3 is None:
    print("One or more images not found. Please check the file names.")
    exit()

# Convert BGR to RGB for display
def show_image(img, title):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Show individual images
show_image(img1, "Image 1")
show_image(img2, "Image 2")
show_image(img3, "Image 3")

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# SIFT feature detection
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
kp3, des3 = sift.detectAndCompute(gray3, None)

# FLANN based matcher
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Match descriptors
matches1to2 = flann.knnMatch(des1, des2, k=2)
matches2to3 = flann.knnMatch(des2, des3, k=2)

# Filter good matches
good_matches1to2 = []
good_matches2to3 = []

for m, n in matches1to2:
    if m.distance < 0.7 * n.distance:
        good_matches1to2.append(m)

for m, n in matches2to3:
    if m.distance < 0.7 * n.distance:
        good_matches2to3.append(m)

# Draw matches
img_matches1to2 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches1to2[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_matches2to3 = cv2.drawMatches(img2, kp2, img3, kp3, good_matches2to3[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show matches
show_image(img_matches1to2, "Matches: Image 1 & 2")
show_image(img_matches2to3, "Matches: Image 2 & 3")

# Find Homographies
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches1to2]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches1to2]).reshape(-1, 1, 2)

pts3 = np.float32([kp2[m.queryIdx].pt for m in good_matches2to3]).reshape(-1, 1, 2)
pts4 = np.float32([kp3[m.trainIdx].pt for m in good_matches2to3]).reshape(-1, 1, 2)

H1, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)
H2, _ = cv2.findHomography(pts3, pts4, cv2.RANSAC)

# Warp and Stitch
height, width = img1.shape[:2]

img2_warped = cv2.warpPerspective(img2, H1, (width * 2, height))
img2_warped[0:height, 0:width] = img1

img3_warped = cv2.warpPerspective(img3, H2, (width * 3, height))
img3_warped[0:height, 0:width * 2] = img2_warped[0:height, 0:width * 2]

# Final stitched image
final_stitched = img3_warped
show_image(final_stitched, "Final Stitched Panorama")
