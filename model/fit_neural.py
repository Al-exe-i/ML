import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Загружаем набор данных Fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Нормализуем данные
train_images = train_images / 255.0
test_images = test_images / 255.0

# Определяем классы одежды
class_names = [
    'Футболка/Топ', 'Брюки', 'Свитер', 'Платье', 'Пальто',
    'Сандалии', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботинки'
]

# Создаем модель нейронной сети
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Преобразуем изображения в 1-мерный массив
    layers.Dense(500, activation='relu'),   # Скрытый слой с 128 нейронами и активацией ReLU
    layers.Dense(10, activation='softmax')   # Выходной слой с 10 нейронами (по количеству классов)
])

# Компилируем модель
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучаем модель
model.fit(train_images, train_labels, epochs=100)

# Оцениваем модель
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nТочность на тестовых данных:', test_acc)

# Предсказываем класс для тестового изображения
predictions = model.predict(test_images)
model.save('clothes_model.h5')

# Пример вывода результата
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.title(f"Предсказание: {class_names[predictions[i].argmax()]}")
    plt.axis('off')
plt.show()
