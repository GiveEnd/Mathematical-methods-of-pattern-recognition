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
