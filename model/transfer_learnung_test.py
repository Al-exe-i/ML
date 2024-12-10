import numpy as np
import tensorflow as tf

cnn_transfer_phones_model = tf.keras.models.load_model('phones_cnn_transfer.keras')

class_names = ['simple', 'smartphone']

img = tf.keras.utils.load_img(
    "hh.jpg", target_size=(160, 160)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = cnn_transfer_phones_model.predict_on_batch(img_array)
predictions = cnn_transfer_phones_model.predict_on_batch(img_array).flatten()
predictions = tf.nn.sigmoid(predictions)
print(predictions)
predictions = tf.where(predictions < 0.6, 0, 1)

print(
    f"Это изображение похоже на {class_names[int(predictions)]}"
)