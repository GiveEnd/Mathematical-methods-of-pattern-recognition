# **Математические методы распознавания образов**


## **Практическое задание**
**Задача:**
 - Установить библиотеку OpenCV
 - Написать программу, выводящую на экран некую картинку
 
**Код:**

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

**Результат:**

 1. Исходное фото
![Исходное фото](https://github.com/GiveEnd/Mathematical-methods-of-pattern-recognition/blob/main/1/Image/1.jpg?raw=true)
 2. Обрезанное фото 
![Обрезанное фото ](https://github.com/GiveEnd/Mathematical-methods-of-pattern-recognition/blob/main/1/Result/Result_2.png?raw=true)

 3. Изменение размера на 20%
![Изменение размера на 20%](https://github.com/GiveEnd/Mathematical-methods-of-pattern-recognition/blob/main/1/Result/Result_3.png?raw=true)

 4. Поворот на 180 градусов
![Поворот на 180 градусов](https://github.com/GiveEnd/Mathematical-methods-of-pattern-recognition/blob/main/1/Result/Result_4.png?raw=true)

 5. В серых цветах
![В серых цветах](https://github.com/GiveEnd/Mathematical-methods-of-pattern-recognition/blob/main/1/Result/Result_5.png?raw=true)

 6. Черное-белое изображение
![Черное-белое изображение](https://github.com/GiveEnd/Mathematical-methods-of-pattern-recognition/blob/main/1/Result/Result_6.png?raw=true)

 7. Блюр
![Блюр](https://github.com/GiveEnd/Mathematical-methods-of-pattern-recognition/blob/main/1/Result/Result_7.png?raw=true)

## **Лабораторная работа №1**

**Задача: разработать программу, детектирующую заданный цвет на сцене**

 - Входной поток поступает с фотографии
 - Выходными данными являются координаты центра объекта, которые выводятся на результирующее изображение рядом с самим объектом
 - (Дополнительно 1) Определять размер объекта в пикселях
 - (Дополнительно 2) Рассмотреть случай детектирования нескольких объектов заданного цвета

**Код:**

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

**Результат:**

![Обнаруженные объекты](https://github.com/GiveEnd/Mathematical-methods-of-pattern-recognition/blob/main/2/Result/Result_1.png?raw=true)

![Маска](https://github.com/GiveEnd/Mathematical-methods-of-pattern-recognition/blob/main/2/Result/Result_2.png?raw=true)

## **Лабораторная работа №2**
**Задача: разработать программу, сшивающую изображения в панораму**
 - Для двух фотографий найти их ключевые точки, описать их, используя любой дескриптор (SIFT, SURF, ORB или др.) и провести их сопоставление. Продемонстрировать результат.
 - Сшить изображения в панораму
 
**Код:**

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
    
**Результат:**

![Совпадения](https://github.com/GiveEnd/Mathematical-methods-of-pattern-recognition/blob/main/3/Result/Result_1.png?raw=true)
![Панорама](https://github.com/GiveEnd/Mathematical-methods-of-pattern-recognition/blob/main/3/Result/Result_2.png?raw=true)

## **Лабораторная работа №3**
**Задача: разработать программу, детектирующую лица людей в фотографии с применением каскадного классификатора.**
 - Выделить на сцене лица, применяя каскадный классификатор из базового набора opencv. Вывести результат.

**Код:**

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

**Результат:**
![1](https://github.com/GiveEnd/Mathematical-methods-of-pattern-recognition/blob/main/4/Result/1.png?raw=true)
![2](https://github.com/GiveEnd/Mathematical-methods-of-pattern-recognition/blob/main/4/Result/2.png?raw=true)
![3](https://github.com/GiveEnd/Mathematical-methods-of-pattern-recognition/blob/main/4/Result/3.png?raw=true)
![4](https://github.com/GiveEnd/Mathematical-methods-of-pattern-recognition/blob/main/4/Result/4.png?raw=true)
![5](https://github.com/GiveEnd/Mathematical-methods-of-pattern-recognition/blob/main/4/Result/5.png?raw=true)

## **Практическое задание**
**Задача:**
 - Сжатие изображения с помощью SVD

**Код:**

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    image = cv2.imread("C:\\Users\\GiveEnd\\Desktop\\Mathematical methods of pattern recognition\\5\\Image\\1.jpg", cv2.IMREAD_GRAYSCALE)
    
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    
    h, w = image.shape
    max_k = min(h, w)
    print(f"Размер изображения: {h} x {w}")
    print(f"Максимальный k = {max_k}")
    
    # сингулярные числа
    k = 3648 
    
    # восстановление приближённого изображения
    reconstructed_image = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
    
    plt.figure(figsize=(10, 5))
    
    # оригинал
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Оригинальное изображение')
    plt.axis('off')
    
    # восстановленное
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f'Приближение с {k} сингулярными числами')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

**Результат:**![1](https://github.com/GiveEnd/Mathematical-methods-of-pattern-recognition/blob/main/5/Result/1.png?raw=true)
![2](https://github.com/GiveEnd/Mathematical-methods-of-pattern-recognition/blob/main/5/Result/2.png?raw=true)
![3](https://github.com/GiveEnd/Mathematical-methods-of-pattern-recognition/blob/main/5/Result/3.png?raw=true)
![4](https://github.com/GiveEnd/Mathematical-methods-of-pattern-recognition/blob/main/5/Result/4.png?raw=true)
![5](https://github.com/GiveEnd/Mathematical-methods-of-pattern-recognition/blob/main/5/Result/5.png?raw=true)
