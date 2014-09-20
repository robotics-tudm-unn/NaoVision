import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt


# === User Constants ===

# The size of chessboard square in meters
scale_factor = 0.0245

# Let chessboard pattern be MxN squares, then
# Number of chessboard nodes by vertical should be: vertical_nodes_num = (M - 1)
vertical_nodes_num = 9
# Number of chessboard nodes by horizontal should be: horizontal_nodes_num = (N - 1)
horizontal_nodes_num = 6

# Pathname to the dir where photos of the chessboard pattern are stored
chessboard_patterns_dirname = 'chessboard_patterns/'

# Pathname for saving camera info
camera_info_dirname = 'camera_info/'

# === Other Constants ===

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((horizontal_nodes_num * vertical_nodes_num, 3), np.float32)
objp[:, :2] = scale_factor * np.mgrid[0:vertical_nodes_num, 0:horizontal_nodes_num].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob(chessboard_patterns_dirname + '/*.jpg')

print images
#print cv2.imread(images[0]).shape
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (vertical_nodes_num,
                                                    horizontal_nodes_num), None)
    # If found, add object points, image points (after refining them)
    if ret is True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (vertical_nodes_num,
                                        horizontal_nodes_num), corners, ret)

        plt.imshow(img)
        plt.show()
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                   (img.shape[1], img.shape[0]), None, None)
print tvecs
# Get optimal camera matrix
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist,
                                                  (gray.shape[1], gray.shape[0]), 1,
                                                  (gray.shape[1], gray.shape[0]))

# Print calibration error
mean_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print "total error: ", mean_error / len(objpoints)

# Save all camera info
np.savetxt(camera_info_dirname + 'mtx.dat', mtx)
np.savetxt(camera_info_dirname + 'dist.dat', dist)
np.savetxt(camera_info_dirname + 'newcameramtx.dat', newcameramtx)
np.savetxt(camera_info_dirname + 'roi.dat', roi)
