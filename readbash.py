import cv2
import numpy as np
import easyocr
import math

def ocr_subregion(subregion, tokenlist):

    # Preprocess the subregion
    gray_subregion = cv2.cvtColor(subregion, cv2.COLOR_BGR2GRAY)
    gray_subregion[0, 0] = 1
    gray_subregion = cv2.normalize(gray_subregion, None, 0, 255, cv2.NORM_MINMAX)
    gray_subregion[0, 0] = 255
    # Display the image
    cv2.imshow('Image', gray_subregion)

    # Wait for a key press
    key = cv2.waitKey(0)

    #return query_openai_with_image(subregion)

    # Perform OCR on the subregion
    #results = reader.readtext(gray_subregion, allowlist=tokenlist)
    results = reader.recognize(gray_subregion, allowlist=tokenlist)
    # Extract the recognized text
    text = ''
    for result in results:
        text += result[1] + ' '

    return text.strip()


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

# Read the image and template
image = cv2.imread('images/tests/bashtest.png')
template = cv2.imread('images/templates/bash2.png')

# Register the image to the template
registered_image = register_image(image, template)


#roi = (240, 300, 33, 67)  # (x, y, width, height)

# Perform OCR on the subregion

# Initialize the EasyOCR reader

allowed_outputs = ["M", "H"]
# "x", "X", "mod", "Mod", "mild", "Mild", "sev", "Sev"]

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
reader = easyocr.Reader(['en'], recog_network='english_g2')
for (curr_y, curr_mth) in ys:
    print(f"{curr_mth}")
    for (curr_x, curr_day) in xs:
        print(f"{curr_mth} {curr_day}")
        roi = (curr_x, curr_y, roi_big_width, roi_height)
        #Extract the region of interest (ROI) from the image
        x, y, w, h = roi
        if curr_day < 10:
            subregion = registered_image[y+pad:y+h-pad, x+pad:x+roi_small_width-pad]
            big_img = 255 * np.ones((roi_height-2*pad, roi_big_width-2*pad, 3), dtype=np.uint8)
            startx=4
            big_img[:, startx:(startx+roi_small_width-2*pad), :] = subregion
            subregion = big_img
        else:
            subregion = registered_image[y+pad:y+h-pad, x+pad:x+roi_big_width-pad]

        text = ocr_subregion(subregion, allowed_outputs)
        print("OCR Result:")
        print(text)
