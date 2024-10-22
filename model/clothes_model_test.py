import numpy as np
from keras.src.utils.module_utils import tensorflow
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

model = tensorflow.keras.models.load_model('clothes_model.h5')

class_names = [
    'Футболка/Топ', 'Брюки', 'Свитер', 'Платье', 'Пальто',
    'Сандалии', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботинки'
]

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')  # Загружаем изображение
    img_array = image.img_to_array(img)  # Преобразуем изображение в массив
    img_array = img_array / 255.0  # Нормализуем
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность
    return img_array


img_path = 'test2.jpg'  # Замените 'path_to_your_image.jpg' на путь к вашему изображению

prepared_image = load_and_preprocess_image(img_path)

predictions = model.predict(prepared_image)
predicted_class = class_names[predictions[0].argmax()]


plt.imshow(image.load_img(img_path, target_size=(28, 28), color_mode='grayscale'))
plt.title(f"Предсказание: {predicted_class}")
plt.axis('off')
plt.show()