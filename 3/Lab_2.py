import cv2
import numpy as np

img1 = cv2.imread("C:\\Users\\GiveEnd\\Desktop\\Mathematical methods of pattern recognition\\3\\Images\\IMG_1392.jpg")
img2 = cv2.imread("C:\\Users\\GiveEnd\\Desktop\\Mathematical methods of pattern recognition\\3\\Images\\IMG_1393.jpg")

orb = cv2.ORB_create()

# ключевые точки и дескрипторы
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# сопоставление дескрипторов
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

# лучшие совпадения (топ 50)
result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
cv2.imshow("Matches", result)

src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

height, width = img1.shape[:2]
new_width = width + img2.shape[1]
result_canvas = cv2.warpPerspective(img2, H, (new_width, height))

result_canvas[0:height, 0:width] = img1

cv2.imshow("Panorama", result_canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
