import pathlib
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np


data_dir = pathlib.Path('../phones_photos/')

batch_size = 32
img_height = 180
img_width = 180

# Обучающий ДС
train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset="training",
                                                       seed=999, image_size=(img_height, img_width),
                                                       batch_size=batch_size)

# Валидационный ДС
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
for i, elem in enumerate(class_names):
    if elem == 'simple':
        class_names[i] = "кнопочный телефон"
    elif elem == 'smartphone':
        class_names[i] = "смартфон"


# Кэширование и параллелизация
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                       img_width,
                                       3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

model = Sequential([
    data_augmentation,
    layers.Rescaling(1. / 255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 35
need_fit_model = True
if need_fit_model:
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    test_loss, test_accuracy = model.evaluate(val_ds)
    print("Доля верных ответов на тестовых данных, в процентах:", round(test_accuracy * 100, 4))
    model.save("phones_cnn.h5")

model = tf.keras.models.load_model('phones_cnn.h5')
img = tf.keras.utils.load_img(
    "fff.png", target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(score)
print(
    f"Это изображение похоже на {class_names[np.argmax(score)]} с вероятностью {100 * np.max(score)} процентов."
)
print(class_names)  # ['simple', 'smartphone']
