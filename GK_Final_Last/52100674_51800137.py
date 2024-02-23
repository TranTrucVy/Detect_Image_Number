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

    #crop anh thanh 4 phan
    img_crop1 = image1[:280, :235, :]
    img_crop2 = image1[280:, :235, :]
    img_crop3 = image1[:280, 235:, :]
    #crop goc duoi phai cua anh thanh 3 phan
    img_crop4 = image1[280:480, 235:, :]
    img_crop5 = image1[480:, 235:460, :]
    img_crop6 = image1[480:, 460:, :]

    #convert anh
    img_gray1 = cv2.cvtColor(img_crop1, cv2.COLOR_BGR2GRAY)
    img_gray2 = cv2.cvtColor(img_crop2, cv2.COLOR_BGR2GRAY)
    img_gray3 = cv2.cvtColor(img_crop3, cv2.COLOR_BGR2GRAY)
    img_gray4 = cv2.cvtColor(img_crop4, cv2.COLOR_BGR2GRAY)
    img_gray5 = cv2.cvtColor(img_crop5, cv2.COLOR_BGR2GRAY)
    img_gray6 = cv2.cvtColor(img_crop6, cv2.COLOR_BGR2GRAY)

    #threshold tung anh
    th1 = cv2.adaptiveThreshold(img_gray1 ,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    th2 = cv2.adaptiveThreshold(img_gray2 ,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img_gray3 ,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    th4 = cv2.adaptiveThreshold(img_gray4 ,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    th5 = cv2.adaptiveThreshold(img_gray5 ,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    th6 = cv2.adaptiveThreshold(img_gray6 ,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

    #kernel xu ly tung anh
    kernel = np.ones((4,4),np.uint8)
    kernel2 = np.ones((2,3),np.uint8)
    kernel3 = np.ones((4,4),np.uint8)
    kernel4 = np.ones((5,4),np.uint8)
    kernel5 = np.ones((5,5),np.uint8)
    kernel6 = np.ones((3,3),np.uint8)

    #xu ly anh
    #anh crop 1
    closing = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel2)
    erosion = cv2.erode(closing,kernel,iterations = 1)
    #anh crop 2
    closing2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel2)
    #anh crop 3
    closing3 = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel2)
    #anh crop 4
    closing4 = cv2.morphologyEx(th4, cv2.MORPH_CLOSE, kernel3)
    #anh crop 5
    closing5 = cv2.morphologyEx(th5, cv2.MORPH_CLOSE, kernel4)
    dilation_image = cv2.dilate(closing5, kernel6, iterations=1)
    #anh crop 6
    closing6 = cv2.morphologyEx(th6, cv2.MORPH_CLOSE, kernel5)  

    #ghep cac phan cua hinh anh 
    img_concat = cv2.hconcat([dilation_image,closing6])
    img_concat2 = cv2.vconcat([closing4,img_concat])
    img_concat3 = cv2.hconcat([erosion,closing3])
    img_concat4 = cv2.hconcat([closing2, img_concat2])
    img_final = cv2.vconcat([img_concat3, img_concat4])

    blurred = cv2.GaussianBlur(img_final, (5, 5), 0)
    
    edged = cv2.Canny(blurred, 30, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 1000 < w * h < 5500 and 23 < w < 200 and h < 100:
            cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite("Output2.png", image1)

if __name__ == "__main__":
    main()
