import cv2
import numpy as np

def recognize():
    img = cv2.imread('resources/originals/triangle.jpg')
    mask = cv2.imread('resources/mask.jpg')

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

    # Print the dimensions and position of the bounding rectangle (for troubleshooting purposes)
    print('Bounding rectangle dimensions:', x, y, w, h)

    # Save the cropped image
    cv2.imwrite('resources/cropped/cropped_triangle.jpg', cropped_img)

    # Display the original image, binary mask, and cropped image (for troubleshooting purposes)
    cv2.imshow('Original image', img)
    cv2.imshow('Binary mask', binary_mask)
    cv2.imshow('Cropped image', cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

