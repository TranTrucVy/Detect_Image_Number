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


def chia_anh_lam_bon(img):
    height, width, _ = img.shape
    chia2partdoc = height // 2
    chia2partngang = width // 2
    top_left = img[0:chia2partdoc, 0:chia2partngang]
    top_right = img[0:chia2partdoc, chia2partngang:width]
    bottom_left = img[chia2partdoc:height, 0:chia2partngang]
    bottom_right = img[chia2partdoc:height, chia2partngang:width]
    return top_left, top_right, bottom_left, bottom_right

def erosion_and_dilation(img, erosion_kernel, dilation_kernel):
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, erosion_kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_kernel)
    erosion_image = cv2.erode(img, kernel1, iterations=1)
    dilation_image = cv2.dilate(erosion_image, kernel2, iterations=1)
    return dilation_image

def process_bottom_right_part(part):
    height_part, width_part, _ = part.shape
    chia_nguyen = height_part // 3

    part1 = part[:chia_nguyen, :]
    part2 = part[chia_nguyen:, :]
    eroi_part1 = erosion_and_dilation(part1, (4, 4), (3, 3))

    height_p2, width_p2, _ = part2.shape
    phan_mo = width_p2 // 2
    phanmonhat = erosion_and_dilation(part2[:, :phan_mo], (6, 6), (2, 6))
    anhcuoicung = erosion_and_dilation(part2[:, phan_mo:], (4, 4), (2, 5))

    part2_after = part2
    part2_after[:, :phan_mo] = phanmonhat
    part2_after[:, phan_mo:] = anhcuoicung

    return np.vstack((eroi_part1, part2_after))

def main():
# 1A
    image = cv2.imread("input1.jpg")

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
# 1B
    threshold_value = 220
    threshold_img = apply_threshold(image, threshold_value)
    cv2.imwrite('Output_1B.jpg', threshold_img)
# 1C
    lower_threshold, upper_threshold = 175, 185
    thresholded_2 = apply_dual_threshold(threshold_img, lower_threshold, upper_threshold)
    cv2.imwrite('Output_1C.jpg', thresholded_2)

#BAITAP2
    image1 = cv2.imread("input2.png")

    binary_image = cv2.threshold(image1, 128, 255, cv2.THRESH_BINARY_INV)[1]
    top_left, top_right, bottom_left, bottom_right = chia_anh_lam_bon(binary_image)

    img_tl = erosion_and_dilation(top_left, (3, 1), (1, 4))
    img_tr = erosion_and_dilation(top_right, (3, 1), (1, 4))
    img_bl = erosion_and_dilation(bottom_left, (3, 1), (3, 3))

    bottom_right_processed = process_bottom_right_part(bottom_right)

    height, width, _ = binary_image.shape
    chia2partdoc = height // 2
    chia2partngang = width // 2
    binary_image[0:chia2partdoc, 0:chia2partngang] = img_tl
    binary_image[0:chia2partdoc, chia2partngang:width] = img_tr
    binary_image[chia2partdoc:height, 0:chia2partngang] = img_bl
    binary_image[chia2partdoc:height, chia2partngang:width] = bottom_right_processed

    blurred = cv2.GaussianBlur(binary_image, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 1000 < w * h < 5500 and 23 < w < 200 and h < 100:
            cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite("Output2.png", image1)

if __name__ == "__main__":
    main()
