import cv2 
import numpy as np

def remove_noise(image_result):
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(image_result, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing

def combine_red_masks(hsv, lower_red1, upper_red1, lower_red2, upper_red2):
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    return mask_red

def process_color(image, lower_bound, upper_bound, color_name):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color_name == 'red':
        mask_color = combine_red_masks(hsv, lower_bound[0], upper_bound[0], lower_bound[1], upper_bound[1])
    else:
        mask_color = cv2.inRange(hsv, lower_bound, upper_bound)
    color_result = cv2.bitwise_and(image, image, mask=mask_color)
    processed_image = remove_noise(color_result)
    cv2.imwrite(f'{color_name}_stars.jpg', processed_image)

def apply_threshold(image, threshold_value):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold_img = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_TOZERO_INV)
    return remove_noise(threshold_img)


def apply_dual_threshold(image, lower_threshold, upper_threshold):
    _, thresholded = cv2.threshold(image, lower_threshold, 255, cv2.THRESH_TOZERO)
    _, thresholded_inv = cv2.threshold(thresholded, upper_threshold, 255, cv2.THRESH_TOZERO_INV)
    return remove_noise(thresholded_inv)

def main():
    # Load the image
    image = cv2.imread("input1.jpg")

    # Define the lower and upper bounds of the colors you want to isolate (in HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([15, 255, 255])

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([120, 255, 255])

    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    lower_purple = np.array([140, 100, 100])
    upper_purple = np.array([160, 255, 255])

    # Process each color and save the result
    color_ranges = {
        'yellow': (lower_yellow, upper_yellow),
        'orange': (lower_orange, upper_orange),
        'red': ((lower_red1, lower_red2), (upper_red1, upper_red2)),
        'blue': (lower_blue, upper_blue),
        'green': (lower_green, upper_green),
        'purple': (lower_purple, upper_purple),
    }

    for i, (color_name, (lower, upper)) in enumerate(color_ranges.items()):
        process_color(image, lower, upper, color_name)
    

    # Apply thresholding to the grayscale image
    threshold_value = 220
    threshold_img = apply_threshold(image, threshold_value)
    cv2.imwrite('Output_1B.jpg', threshold_img)


    # Apply dual thresholding to separate the stars from the background for the red color
    lower_threshold, upper_threshold = 175, 185
    thresholded_2 = apply_dual_threshold(threshold_img, lower_threshold, upper_threshold)
    cv2.imwrite('Output_1C.jpg', thresholded_2)

if __name__ == "__main__":
    main()
