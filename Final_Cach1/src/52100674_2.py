import cv2
import numpy as np

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
