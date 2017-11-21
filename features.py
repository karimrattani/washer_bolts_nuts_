import cv2
import numpy as np
def get_object_area_ratio(img):
    box_area=img.shape[0]*img.shape[1]
    object_area=np.count_nonzero(img)
    ratio=object_area/box_area
    return ratio
def get_boundary_shape(img):
    height=img.shape[0]
    width=img.shape[1]
    if(width<=height):
        return width/height
    else:
        return height/width
def get_corner(img):
    dst=cv2.cornerHarris(img,2,3,0.04)
    corners = np.count_nonzero(dst[dst > 0.01 * dst.max()])
    return corners,dst

def get_circularity_error(img):
    b, object_conts, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_areas = [cv2.contourArea(object_conts[i]) for i in range(len(object_conts))]
    max_contour_idx = np.argmax(contour_areas)
    area_object = contour_areas[max_contour_idx]
    radius=0
    if img.shape[0]>img.shape[1]:
        radius=img.shape[0] / 2.0
    else:
        radius=img.shape[1] / 2.0
    area_circle = np.pi * radius ** 2

    circularity_error = abs(area_object - area_circle) / ((2 * radius) ** 2)
    return circularity_error

def get_features(img):
    a_ratio = get_object_area_ratio(img)
    box_scale = get_boundary_shape(img)
    total_corners, corner_img = get_corner(img)
    area=img.shape[0]*img.shape[1]
    cornerPerArea=total_corners*100/area
    circularity=get_circularity_error(img)
    return [a_ratio, box_scale, total_corners,circularity,cornerPerArea]