"""
    Before using:
    Define the path to the folder where camera info information is stored
    with camera_info_dir variable.
    To change the drawing image look to main() function.
"""
import cv2
import numpy as np


# ====================================
# User's Constants
# ====================================

# Dirname of camera information files
camera_info_dir = 'camera_info/'

# Coordinates of paper_sheet vertexes in meters, that are determinated in paper_sheet's
# coordinate system.
paper_sheet_vertexes = np.array([[0., 0., 0.],
                                [0.1485, 0., 0.],
                                [0.1485, 0.21, 0.],
                                [0., 0.21, 0.]], np.float32)

# The stopping criteria of cornerSubPix() method.
cornerSubPix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ====================================
# Constants
# ====================================

newcameramtx = np.loadtxt(camera_info_dir + 'newcameramtx.dat')
roi = np.loadtxt(camera_info_dir + 'roi.dat')
cam_intr_mtx = np.loadtxt(camera_info_dir + 'mtx.dat')
cam_dist = np.loadtxt(camera_info_dir + 'dist.dat')

# ====================================


def organize_vertexes(vertexes):
    """
        Organize vertexes in the order, that is needed for finding
        projection transformation.
        Input:
            vertexes - list of not organized 2D coordinates
        Output:
            result_array - 4x2 numpy array of organized vertexes coordinates,
                where 4 for the number of vertexes
                and 2 for the number of coordinates on the picture.

    """
    result_list = list()

    # Find two upper vertexes and rate them.
    min_y = np.sort(vertexes[:, 1])[1]  # magic!
    a, b = np.where(vertexes[:, 1] <= min_y)[0]

    if vertexes[a, 0] < vertexes[b, 0]:
        result_list += [vertexes[a, :], vertexes[b, :]]
    else:
        result_list += [vertexes[b, :], vertexes[a, :]]

    # Find two lower vertexes and rate them.
    a, b = [item for item in range(4) if item not in [a, b]]
    if vertexes[a, 0] > vertexes[b, 0]:
        result_list += [vertexes[a, :], vertexes[b, :]]
    else:
        result_list += [vertexes[b, :], vertexes[a, :]]

    result_array = np.array(result_list)
    return result_array


def find_paper_sheet_in_contours(contours_list):
    """
        Find the most likely paper_sheet contour in all contours list.
        Input:
            contours_list - list of all found by cv2.findContours() contours.
        Output:
            is_any_detected - the flag that shows is any appropriate
                contours are found.
            paper_sheet_crds - numpy array of 2D coordinates of paper_sheet's
                vertexes on picture in pxls.
    """

    is_any_detected = False
    trapezoid_contours = []
    for contour in contours_list:
        # Parameter specifying the approximation accurancy. This is the maximum
        # distance between the original curve
        # and its approximation.
        approx_epsilon = 0.1 * cv2.arcLength(contour, True)

        # Approximate contour curve using
        # Ramer-Douglas-Peucker algorithm
        approx = cv2.approxPolyDP(contour, approx_epsilon, True)

        # Filter the contours
        if (len(approx) == 4) \
            and (abs(cv2.contourArea(approx)) > 500) \
                and cv2.isContourConvex(approx):
            trapezoid_contours.append(approx)

    # if any trapezoids detected:
    if len(trapezoid_contours) > 0:
        # As the result paper_sheet contour we will take one with the largest area.
        area_list = map(lambda x: abs(cv2.contourArea(x)), trapezoid_contours)
        list_max = max(area_list)
        max_pos = [i for i, j in enumerate(area_list) if j == list_max][0]
        paper_sheet_contour = trapezoid_contours[max_pos]

        # Convert opencv contour to the list of formed dots with stupid magic
        # and organize them
        paper_sheet_crds = np.array(map(lambda x: paper_sheet_contour[x][0], range(4)), np.float32)
        paper_sheet_crds = organize_vertexes(paper_sheet_crds)
        is_any_detected = True
    else:
        paper_sheet_contour = []
        paper_sheet_crds = None

    return is_any_detected, paper_sheet_crds


def detect_paper_sheet(src_img):
    """
        Detect the paper_sheet on the picture and return its coordinates in pxls
        Input:
            src_img - BGR image
        Output:
            is_detected - is any paper_sheet detected
            paper_sheet_crds - numpy array of 2D coordinates of paper_sheet's
                vertexes on picture in pxls.
            cmr_rvec - the paper sheet coordinate system rotation
                as seemes in camera coordinate system
            cmr_tvec - the transition from camera coordinate system center
                towards the paper sheet coordinate system center
    """

    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    vertexes = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    all_contours, hierarchy = cv2.findContours(vertexes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    is_detected, paper_sheet_crds = find_paper_sheet_in_contours(all_contours)

    # If any paper_sheet found, then find the coordinates of vertexes with subpixel
    # preciese and draw the found contour on the image
    if is_detected:
        cv2.cornerSubPix(gray_img, paper_sheet_crds, (11, 11), (-1, -1), cornerSubPix_criteria)
        ret_val, cmr_rvec, cmr_tvec = cv2.solvePnP(paper_sheet_vertexes,
                                                   paper_sheet_crds, cam_intr_mtx, cam_dist)
    else:
        cmr_rvec = None
        cmr_tvec = None

    return is_detected, paper_sheet_crds, cmr_rvec, cmr_tvec


def plot_projected_img(src_img, background_img, output_crds):
    """
        Find the projection into found paper_sheet and draw projected image on it.
        Input:
            src_img - the image that will be projected
            background_img - the image where the src_img will be projected to
            output_crds - found pxl coordinates of paper_sheet on the image
        Output:
            rst_img - result image with drawn paper_sheet contour
    """

    # Coordinates of src_img vertexes in its own pxl coordinate system
    src_crds = np.array([[0, 0],
                        [src_img.shape[1], 0],
                        [src_img.shape[1], src_img.shape[0]],
                        [0., src_img.shape[0]]], np.float32)
    background_img = np.array(background_img).copy()
    projection_mtx = cv2.getPerspectiveTransform(src_crds, output_crds)
    background_img_size = background_img.shape[:2]

    # Project src_img
    dst = cv2.warpPerspective(src_img, projection_mtx, background_img_size[::-1],
                              borderValue=(255, 255, 255))

    # Draw projected src_img on background_img
    rst_img = cv2.bitwise_and(background_img, dst)

    return rst_img


def draw_on_found_paper_sheet(drawing):
    """
        The main function.
        Input:
            drawing - BGR image that need to be drawn
    """

    # Initialize video client
    cv2.namedWindow("find_paper_sheet")
    vc = cv2.VideoCapture(0)

    # Try to get the first frame
    if vc.isOpened():
        rval, tmp_frame = vc.read()
        # Undistort frame
        frame = cv2.undistort(tmp_frame, cam_intr_mtx, cam_dist, None, newcameramtx)
    else:
        rval = False

    while rval:
        # Detect the paper sheet
        is_detected, paper_vertexes, cmr_rvec, cmr_tvec = detect_paper_sheet(frame)

        # Draw everything if paper sheet is detected
        if is_detected:
            # Draw projected drawing on the paper sheet
            projected_drawing = plot_projected_img(drawing, frame, paper_vertexes)

            # Draw the boundaries of the paper sheet
            # cv2.drawContours(frame, np.array([convert_dots_list2cv_contour(paper_vertexes)], dtype='int32'), -1, (0, 255, 0), 3)

            # Write distance between the camera and the paper sheet coordinate
            # systems origins
            cv2.putText(projected_drawing,
                        'x:%.3fm|y:%.3fm|z:%.3fm' % (cmr_tvec[0], cmr_tvec[1], cmr_tvec[2]),
                        (0, 20),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 0, 0))

            # Mark the paper sheet coordinate system origin with red
            cv2.circle(projected_drawing, (paper_vertexes[0, 0], paper_vertexes[0, 1]), 5, (0, 0, 255), -1)

            # Mark other paper sheet vertexes dots with green
            for ind in xrange(3):
                cv2.circle(projected_drawing, (paper_vertexes[ind + 1, 0], paper_vertexes[ind + 1, 1]), 5, (0, 255, 0), -1)

            # Crop the image
            x, y, w, h = roi
            projected_drawing = projected_drawing[y: y + h, x: x + w]

            cv2.imshow("find_paper_sheet", projected_drawing)
        else:
            # Crop the image
            x, y, w, h = roi
            frame = frame[y: y + h, x: x + w]

            cv2.imshow("find_paper_sheet", frame)

        rval, tmp_frame = vc.read()
        # Undistort frame
        frame = cv2.undistort(tmp_frame, cam_intr_mtx, cam_dist, None, newcameramtx)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow("find_paper_sheet")
    pass


def main():
    drawing = cv2.imread('test.png')
    draw_on_found_paper_sheet(drawing)
    pass

if __name__ == "__main__":
    main()
