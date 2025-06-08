import cv2

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image = cv2.imread("C:\\Users\\GiveEnd\\Desktop\\Mathematical methods of pattern recognition\\1\\Image\\1.jpg")

cropped = image[1500:2000, 3000:4000]

scale_percent = 10
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

(h, w, d) = image.shape
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 180, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret,threshold_image = cv2.threshold(gray_image, 127, 255, 0)

blurred = cv2.GaussianBlur(image, (51, 51), 0)



viewImage(image, "Original")
viewImage(cropped, "Cropped")
viewImage(resized, "After change a size on 20 %")
viewImage(rotated, "After turning 180 degrees")
viewImage(gray_image, "In grayscale")
viewImage(threshold_image, "Black and white")
viewImage(blurred, "Blur")