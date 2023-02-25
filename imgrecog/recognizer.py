import math

import cv2
import numpy as np


def recognize(mask, img, number):
    # Convert the mask to grayscale
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the grayscale mask
    _, binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)

    # Find the contour of the triangle in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    triangle_contour = max(contours, key=cv2.contourArea)

    # Extract the bounding rectangle of the triangle
    x, y, w, h = cv2.boundingRect(triangle_contour)

    # Crop the binary mask to the dimensions of the bounding rectangle
    cropped_mask = binary_mask[y:y + h, x:x + w].astype(np.uint8)

    # Convert the cropped mask to grayscale
    cropped_mask = cv2.cvtColor(cropped_mask, cv2.COLOR_GRAY2BGR)

    # Perform template matching to locate the triangle in the original image
    result = cv2.matchTemplate(img, cropped_mask, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Extract the coordinates of the top-left corner of the template
    top_left = max_loc

    # Create a rectangle using the template dimensions
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

    # Crop the original image to the dimensions of the template
    cropped_img = img[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]

    # Save the cropped image
    cv2.imwrite('resources/cropped/triangle_cropped_' + str(number) + '.jpg', cropped_img)

    # Display the original image, binary mask, and cropped image (for troubleshooting purposes)
    cv2.imshow('Original image', img)
    cv2.imshow('Binary mask', binary_mask)
    cv2.imshow('Cropped image', cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def recognize_rotated(mask, img, number):
    # Convert the mask to grayscale
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the grayscale mask
    _, binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)

    # Find the contour of the triangle in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    triangle_contour = max(contours, key=cv2.contourArea)

    # Extract the bounding rectangle of the triangle
    x, y, w, h = cv2.boundingRect(triangle_contour)

    # Crop the binary mask to the dimensions of the bounding rectangle
    cropped_mask = binary_mask[y:y + h, x:x + w].astype(np.uint8)

    # Convert the cropped mask to grayscale
    cropped_mask = cv2.cvtColor(cropped_mask, cv2.COLOR_GRAY2BGR)

    # Perform template matching to locate the triangle in the original image
    result = cv2.matchTemplate(img, cropped_mask, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Extract the coordinates of the top-left corner of the template
    top_left = max_loc

    # Create a rectangle using the template dimensions
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

    # Compute the homography matrix between the mask image and the cropped image
    M, _ = cv2.findHomography(triangle_contour[:, 0, :], np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32))

    # Warp the cropped image to align with the mask image
    warped_cropped_img = cv2.warpPerspective(img[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w], M, (w, h))

    # Save the cropped image
    cv2.imwrite('resources/cropped/triangle_cropped_' + str(number) + '.jpg', warped_cropped_img)

    # Display the original image, binary mask, and warped cropped image (for troubleshooting purposes)
    cv2.imshow('Original image', img)
    cv2.imshow('Binary mask', binary_mask)
    cv2.imshow('Warped cropped image', warped_cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def recognize_rotated2(mask, img, number):
    # Convert the mask to grayscale
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the grayscale mask
    _, binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)

    # Find the contour of the triangle in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    triangle_contour = max(contours, key=cv2.contourArea)

    # Extract the bounding rectangle of the triangle
    x, y, w, h = cv2.boundingRect(triangle_contour)

    # Crop the binary mask to the dimensions of the bounding rectangle
    cropped_mask = binary_mask[y:y + h, x:x + w].astype(np.uint8)

    # Convert the cropped mask to grayscale
    cropped_mask = cv2.cvtColor(cropped_mask, cv2.COLOR_GRAY2BGR)

    # Perform template matching to locate the triangle in the original image
    result = cv2.matchTemplate(img, cropped_mask, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Extract the coordinates of the top-left corner of the template
    top_left = max_loc

    # Create a rectangle using the template dimensions
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

    # Crop the original image to the dimensions of the template
    cropped_img = img[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]

    # Rotate the cropped image
    min_rect = cv2.minAreaRect(triangle_contour)
    angle_diff = min_rect[2] - 90.0  # 90 degrees difference due to original triangle shape
    rows, cols = cropped_img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle_diff, 1)
    rotated_img = cv2.warpAffine(cropped_img, M, (cols, rows))

    # Save the cropped and rotated image
    cv2.imwrite('resources/cropped/triangle_cropped_' + str(number) + '.jpg', rotated_img)

    # Display the original image, binary mask, and cropped image (for troubleshooting purposes)
    cv2.imshow('Original image', img)
    cv2.imshow('Binary mask', binary_mask)
    cv2.imshow('Cropped and rotated image', rotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def recognize_rotated3(mask, img, number):
    # Convert the mask to grayscale
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the grayscale mask
    _, binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)

    # Find the contour of the triangle in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    triangle_contour = max(contours, key=cv2.contourArea)

    # Extract the bounding rectangle of the triangle
    x, y, w, h = cv2.boundingRect(triangle_contour)

    # Crop the binary mask to the dimensions of the bounding rectangle
    cropped_mask = binary_mask[y:y + h, x:x + w].astype(np.uint8)

    # Convert the cropped mask to grayscale
    cropped_mask = cv2.cvtColor(cropped_mask, cv2.COLOR_GRAY2BGR)

    # Perform template matching to locate the triangle in the original image
    result = cv2.matchTemplate(img, cropped_mask, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Extract the coordinates of the top-left corner of the template
    top_left = max_loc

    # Create a rectangle using the template dimensions
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

    # Crop the original image to the dimensions of the template
    cropped_img = img[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]

    # Check the rotation of the mask and rotate the cropped image accordingly
    rotation_angle = get_rotation_angle(mask, cropped_mask)
    if rotation_angle != 0:
        h, w = cropped_img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
        cropped_img = cv2.warpAffine(cropped_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Save the cropped image
    cv2.imwrite('resources/cropped/triangle_cropped_' + str(number) + '.jpg', cropped_img)

    # Display the original image, binary mask, and cropped image (for troubleshooting purposes)
    cv2.imshow('Original image', img)
    cv2.imshow('Binary mask', binary_mask)
    cv2.imshow('Cropped image', cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_rotation_angle(mask, img):
    # Convert the mask and input images to grayscale
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the grayscale images
    _, binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)
    _, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

    # Find the contours in the binary images
    mask_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour in each image
    mask_contour = max(mask_contours, key=cv2.contourArea)
    img_contour = max(img_contours, key=cv2.contourArea)

    # Find the minimum area rectangles that bound the contours
    mask_rect = cv2.minAreaRect(mask_contour)
    img_rect = cv2.minAreaRect(img_contour)

    # Find the angles of rotation of the rectangles
    mask_angle = mask_rect[-1]
    img_angle = img_rect[-1]

    return mask_angle - img_angle
