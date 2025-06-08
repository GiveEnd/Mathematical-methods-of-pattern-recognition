import cv2
import numpy as np

image = cv2.imread("C:\\Users\\GiveEnd\\Desktop\\Mathematical methods of pattern recognition\\2\\Image\\1.jpg")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Диапазон цвета (красный, зеленый)
lower_color1 = np.array([170, 100, 100])
upper_color1 = np.array([179, 255, 255])

lower_red2 = np.array([72, 100, 150])
upper_red2 = np.array([92, 255, 255])


mask1 = cv2.inRange(hsv, lower_color1, upper_color1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

masks = cv2.bitwise_or(mask1, mask2)

contours, _ = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 5, (255, 0, 0), -1)

        text = f"X:{cX}, Y:{cY}, Area:{int(area)} px"
        cv2.putText(image, text, (cX + 10, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

cv2.imshow("Detected Objects", image)
cv2.imshow("Mask", masks)

cv2.waitKey(0)
cv2.destroyAllWindows()
