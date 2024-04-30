import cv2
import numpy as np
import torch


def display_im(subregion):

    # Preprocess the subregion
    gray_subregion = cv2.cvtColor(subregion, cv2.COLOR_BGR2GRAY)
    gray_subregion[0, 0] = np.min(gray_subregion)
    gray_subregion = cv2.normalize(gray_subregion, None, 0, 255, cv2.NORM_MINMAX)
    gray_subregion[0, 0] = 255
    # Display the image
    cv2.imshow('Image', gray_subregion)
    key = cv2.waitKey(0)



def register_image(image, template):
    # Convert the images to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors using SIFT
    sift = cv2.SIFT_create()
    keypoints_image, descriptors_image = sift.detectAndCompute(gray_image, None)
    keypoints_template, descriptors_template = sift.detectAndCompute(gray_template, None)

    # Match the descriptors using FLANN matcher
    flann_matcher = cv2.FlannBasedMatcher()
    matches = flann_matcher.knnMatch(descriptors_image, descriptors_template, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Extract the matched keypoints
    src_pts = np.float32([keypoints_image[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_template[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate the transformation matrix using RANSAC
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Apply the transformation to the image
    registered_image = cv2.warpPerspective(image, M, (template.shape[1], template.shape[0]))

    return registered_image


def get_image_subregion_list(image):
    # Read the image and template
    image = cv2.imread(image)
    template = cv2.imread('images/templates/bash2.png')

    # Register the image to the template
    registered_image = register_image(image, template)
    subregion_list = []

    roi_height = 67
    y0 = 299
    xs = [(240), 273, 305, 338, 371, 403, 436, 469, 501, 534, 576, 618, 661, 703, 745, 786, 830, 872, 914, 956, 999, 1041, 1083, 1125, 1168, 1210, 1252, 1294, 1337, 1378, 1421]
    xs = list(zip(xs, range(1, 32)))
    ys = [300+roi_height*(i-1) for i in range(1, 7)]
    mths = ["Jan", "Feb", "Mar", "April", "May", "June"]

    ys = list(zip(ys, mths))

    roi_small_width = 33
    roi_big_width = 42
    pad = 3

    for (curr_y, curr_mth) in ys:
        for (curr_x, curr_day) in xs:
            roi = (curr_x, curr_y, roi_big_width, roi_height)
            #Extract the region of interest (ROI) from the image
            x, y, w, h = roi
            if curr_day < 10:
                subregion = registered_image[y+pad:y+h-pad, x+pad:x+roi_small_width-pad]
                big_img = 255 * np.ones((roi_height-2*pad, roi_big_width-2*pad, 3), dtype=np.uint8)
                startx=4
                big_img[:, startx:(startx+roi_small_width-2*pad), :] = subregion
                
                #fix smaller boxes
                for y2 in range(0, roi_height-2*pad):
                    for x2 in range(0, startx):
                        big_img[y2, x2, :] = registered_image[y+y2+pad, x + pad, :]
                
                for y2 in range(0, roi_height-2*pad):
                    for x2 in range(startx+roi_small_width-2*pad-1, roi_big_width-2*pad):
                        big_img[y2, x2, :] = registered_image[y+y2+pad, x + roi_small_width - pad - 2, :]

                subregion = big_img
            else:
                subregion = registered_image[y+pad:y+h-pad, x+pad:x+roi_big_width-pad]

            subregion_list.append((curr_y, curr_mth, subregion))
    return subregion_list
