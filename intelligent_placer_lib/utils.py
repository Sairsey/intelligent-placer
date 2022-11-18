import numpy as np
import cv2
from shapely import geometry, ops

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def rotate_contour(cnt, angle):
    M = cv2.moments(cnt)
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']

    cnt_norm = cnt - [cx, cy]

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    return cnt_rotated

def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = M['m10']/M['m00']
    cy = M['m01']/M['m00']

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]

    return cnt_scaled

def contours_intersect_area(contour1,contour2):
    polygon1 = [(el[0][0], el[0][1]) for el in contour1]
    polygon2 = [(el[0][0], el[0][1]) for el in contour2]
    polygon1 = geometry.Polygon(polygon1)
    polygon2 = geometry.Polygon(polygon2)

    intersect = polygon1.intersection(polygon2)
    return max(intersect.area, 0)

# Ray tracing
def point_in_contour(x,y,poly):

    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

def contours_distance(contour1,contour2):
    cur_dist = float("inf")
    polygon = [(el2[0][0], el2[0][1]) for el2 in contour2]
    for el1 in contour1:
        point1 = (el1[0][0], el1[0][1])

        if (point_in_contour(point1[0], point1[1], polygon)):
            return 0

        for el2 in contour2:
            point2 = (el2[0][0], el2[0][1])
            vec = (point1[0] - point2[0], point1[1] - point2[1])
            cur_dist = min(cur_dist, np.sqrt(vec[0] ** 2 + vec[1] ** 2))
    return cur_dist

def generate_colors_image(dominant_colors):
    dominant_colors_image = np.zeros((128, 128 * len(dominant_colors), 3))
    for i in range(len(dominant_colors)):
        color = (dominant_colors[i][1][2] / 255, dominant_colors[i][1][1] / 255, dominant_colors[i][1][0] / 255)
        dominant_colors_image = cv2.rectangle(dominant_colors_image, (128 * i, 0), (128 * (i + 1), 128), color, -1)
    return dominant_colors_image

def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v

def normalize_image(img):
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return result_norm