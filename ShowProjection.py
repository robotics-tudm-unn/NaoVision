"""
! Before using:
Put your camera info text files into 'camera_info/' folder.
"""
import cv2
import numpy as np

sheet_A5 = np.array([[0., 0., 0.],
                    [0.1485, 0., 0.],
                    [0.1485, 0.21, 0.],
                    [0., 0.21, 0.]], np.float32)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

class SheetPosition:

    pass


def convert_dots_list2cv_contour(dots_list):
    contour = map(lambda x: [x], dots_list)
    return contour


def organize_edges(edges):
    """
        Organize vertexes in the order, that is needed for finding projection transformation.
    """
    result_list = list()
    # Find two upper vertexes and rate them.
    min_y = np.sort(edges[:, 1])[1]  # magic!
    a, b = np.where(edges[:, 1] <= min_y)[0]

    if edges[a, 0] < edges[b, 0]:
        result_list += [edges[a, :], edges[b, :]]
    else:
        result_list += [edges[b, :], edges[a, :]]

    # Find two lower vertexes and rate them.
    a, b = [item for item in range(4) if item not in [a, b]]
    if edges[a, 0] > edges[b, 0]:
        result_list += [edges[a, :], edges[b, :]]
    else:
        result_list += [edges[b, :], edges[a, :]]

    result_array = np.array(result_list)
    return result_array


def find_sheet_in_contours(contours_list):
    """
        Find the most likely sheet contour in all contours list.
    """
    is_any_detected = False
    trapezoid_contours = []
    for contour in contours_list:
        # print 'sheet_cntr = ', contour
        approx_epsilon = 0.1 * cv2.arcLength(contour, True)     # Parameter specifying the approximation accuracy.
                                                                # This is the maximum distance between the original curve
                                                                # and its approximation.
        approx = cv2.approxPolyDP(contour, approx_epsilon, True)    # Approximate contour curve using
                                                                    # Ramer-Douglas-Peucker algorithm
        if (len(approx) == 4) \
            and (abs(cv2.contourArea(approx)) > 500) \
                and cv2.isContourConvex(approx):                # Filter the contours

            trapezoid_contours.append(approx)

    # if any trapezoids detected:
    if len(trapezoid_contours) > 0:
        # As the result sheet contour we will take one with the largest area.
        area_list = map(lambda x: abs(cv2.contourArea(x)), trapezoid_contours)
        list_max = max(area_list)
        max_pos = [i for i, j in enumerate(area_list) if j == list_max][0]
        sheet_contour = trapezoid_contours[max_pos]

        sheet_crds = np.array([sheet_contour[3][0], sheet_contour[0][0],
                               sheet_contour[1][0], sheet_contour[2][0]], np.float32)
        sheet_crds = organize_edges(sheet_crds)
        is_any_detected = True
    else:
        sheet_contour = []
        sheet_crds = None

    return is_any_detected, sheet_crds


def detect_sheet(src_img):
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    all_contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    is_detected, sheet_crds = find_sheet_in_contours(all_contours)
    if is_detected:
        cv2.cornerSubPix(gray_img, sheet_crds, (11, 11), (-1, -1), criteria)
        cv2.drawContours(src_img, np.array([convert_dots_list2cv_contour(sheet_crds)], dtype='int32'),
                         -1, (0, 255, 0), 3)
    return is_detected, sheet_crds


def get_sheet_contour(contour_list):
    """
    Function prototype
    """
    return contour_list[0]


def plot_projected_img(src_img, background_img, src_crds, output_crds):
    background_img = np.array(background_img)
    projection_mtx = cv2.getPerspectiveTransform(src_crds, output_crds)
    tmp_size = background_img.shape[:2]
    dst = cv2.warpPerspective(src_img, projection_mtx, tmp_size[::-1],
                              borderValue=(255, 255, 255))  # , background_img.shape
    rst_img = cv2.bitwise_and(background_img, dst)
    # rst_img = dst
    return rst_img




def main():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    drawing = cv2.imread('test.png')


    pxl_sheet_size = np.array([[0, 0],
                            [drawing.shape[1], 0],
                            [drawing.shape[1], drawing.shape[0]],
                            [0., drawing.shape[0]]], np.float32)

    cam_intr_mtx = np.loadtxt('camera_info/mtx.dat')
    cam_dist = np.loadtxt('camera_info/dist.dat')


    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:

        is_detected, edges = detect_sheet(frame)

        # is_detected = False
        if is_detected:
            projected_drawing = plot_projected_img(drawing, frame, pxl_sheet_size, edges)
            # cv2.imshow("preview", edges)
            rvec, cmr_rvec, cmr_tvec = cv2.solvePnP(sheet_A5, edges, cam_intr_mtx, cam_dist)
            cv2.putText(projected_drawing,
                        'x:%.3fm|y:%.3fm|z:%.3fm' % (cmr_tvec[0], cmr_tvec[1], cmr_tvec[2]),
                        (0, 20),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 0, 0))
            cv2.circle(projected_drawing, (edges[0, 0], edges[0, 1]), 5, (0, 0, 255), -1)
            cv2.imshow("preview", projected_drawing)
        else:
            cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:   # exit on ESC
            break
    cv2.destroyWindow("preview")

if __name__ == "__main__":
    main()
