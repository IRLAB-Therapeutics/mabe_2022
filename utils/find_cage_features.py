import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage.filters import gaussian


def find_edge(contrast_image, is_horizontal=True):
    tested_angles = np.linspace(-np.pi / 60, np.pi / 60, 300)
    if is_horizontal:
        tested_angles += np.pi/2

    contrast_image -= contrast_image.min()
    contrast_image /= contrast_image.max()
    contrast_image = contrast_image > 0.8

    h, theta, d = hough_line(contrast_image, theta=tested_angles)
    hpeaks = hough_line_peaks(h, theta, d, threshold=0.9*h.max(), min_distance=100,  min_angle=50)
    # if len(hpeaks[0]) > 1:
    #     print("Warning: Found many lines")
    #     print(hpeaks)
    angle, dist = hpeaks[1][0], hpeaks[2][0]
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    slope=np.tan(angle + np.pi/2)
    return (x0,y0), slope


def find_intersection(edge_0, edge_1):
    (x0,y0), slope0 = edge_0
    (x1,y1), slope1 = edge_1

    a0 = y0 - slope0 * x0
    a1 = y1 - slope1 * x1

    x_intersection = (a1 - a0) / (slope0 - slope1)
    y_intersection = y0 + slope0 * (x_intersection - x0)

    return x_intersection, y_intersection


def find_corners(frame):
    smoothed_frame = gaussian(frame, sigma=10)

    bottom_edge_contrast = smoothed_frame[:-1,:,0] - smoothed_frame[1:,:,0]
    top_edge_contrast = -bottom_edge_contrast
    right_edge_contrast = smoothed_frame[:,:-1,0] - smoothed_frame[:,1:,0]
    left_edge_contrast = - right_edge_contrast
    edges = [find_edge(contrast_image, is_horizontal) for contrast_image, is_horizontal in zip(
        [top_edge_contrast, bottom_edge_contrast, left_edge_contrast, right_edge_contrast],
        [True, True, False, False]
    )]
    [top_edge, bottom_edge, left_edge, right_edge] = edges
    top_left = find_intersection(top_edge, left_edge)
    top_right = find_intersection(top_edge, right_edge)
    bottom_left = find_intersection(bottom_edge, left_edge)
    bottom_right = find_intersection(bottom_edge, right_edge)
    return [top_left, top_right, bottom_left, bottom_right]


def get_feeding_location(top_left, top_right):
    mid_point = (np.array(top_left) + top_right) / 2
    return mid_point[0], mid_point[1] - 20