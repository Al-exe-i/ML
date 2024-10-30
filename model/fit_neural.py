import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import numpy as np

# Загружаем набор данных Fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Определяем классы одежды
class_names = [
    'Футболка/Топ', 'Брюки', 'Свитер', 'Платье', 'Пальто',
    'Сандалии', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботинки'
]

# Нормализуем данные
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train_cat = utils.to_categorical(y_train, 10)
y_test_cat = utils.to_categorical(y_test, 10)

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)



# Создаем модель нейронной сети
model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10,  activation='softmax')
])

# Компилируем модель
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Обучаем модель
model.fit(x_train, y_train_cat, batch_size=32, epochs=8, validation_split=0.2)

# Оцениваем модель
scores = model.evaluate(x_test, y_test_cat)
print("Доля верных ответов на тестовых данных, в процентах:", round(scores[1] * 100, 4))

# Предсказываем класс для тестового изображения
predictions = model.predict(x_test)
model.save('clothes_model.h5')

# Пример вывода результата
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.title(f"Предсказание: {class_names[predictions[i].argmax()]}")
    plt.axis('off')
plt.show()
