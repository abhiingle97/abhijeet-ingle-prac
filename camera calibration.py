import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a local chessboard image
image_path = "chessboard.jpg"
img = cv2.imread(image_path)

if img is None:
    print("Image not found. Please place 'chessboard.jpg' in the same folder.")
    exit()

# Resize image for faster processing
img = cv2.resize(img, (640, 480))

# Chessboard pattern size (number of internal corners)
chessboard_size = (9, 6)
square_size = 1.0  # Any unit (e.g., 1cm or 1inch)

# Prepare 3D object points in real-world space
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2) * square_size

obj_points = []  # 3D real-world points
img_points = []  # 2D image points

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect chessboard corners
print("Starting chessboard detection...")
ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
print(f"Chessboard detection result: {ret}")

# If found, proceed
if ret:
    obj_points.append(objp)
    img_points.append(corners)

    # Draw corners
    img = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Chessboard Corners")
    plt.axis('off')
    plt.show()
else:
    print("Chessboard corners not found.")
    exit()

# Camera Calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None)

# Print camera parameters
print("\nCamera matrix:\n", camera_matrix)
print("\nDistortion coefficients:\n", dist_coeffs)

# Undistort image
h, w = img.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

# Display original vs undistorted
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
plt.title("Undistorted Image")
plt.axis('off')

plt.show()
