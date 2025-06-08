import cv2
import os

folder_path = "C:\\Users\\GiveEnd\\Desktop\\Mathematical methods of pattern recognition\\4\\Image"

# несколько каскадных классификаторов для лиц
cascade1 = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cascade2 = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
cascade3 = cv2.CascadeClassifier('haarcascade_profileface.xml')

# список изображений
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg'))]

for image_name in image_files:
    image_path = os.path.join(folder_path, image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Не удалось открыть {image_name}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces1 = cascade1.detectMultiScale(gray, 1.1, 5)
    faces2 = cascade2.detectMultiScale(gray, 1.1, 5)
    faces3 = cascade3.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces1:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # зелёный

    for (x, y, w, h) in faces2:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # синий

    for (x, y, w, h) in faces3:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # красный

    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()