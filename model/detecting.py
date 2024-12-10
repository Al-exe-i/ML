import torch
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
image = Image.open('test.png')
results = model(image)
#results.show()
# Список классов, которые нас интересуют
desired_classes = ['handbag', 'car']
# Порог вероятности
confidence_threshold = 0.5
# Фильтрация результатов по классам и вероятности
filtered_results = results.pandas().xyxy[0]
filtered_results = filtered_results[(filtered_results['name'].isin(desired_classes)) & (filtered_results['confidence'] >= confidence_threshold)]
filtered_image = np.array(image)
for _, row in filtered_results.iterrows():
    label = row['name']
    conf = row['confidence']
    xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
    color = (0, 255, 0)  # Зеленый цвет для рамки
    cv2.rectangle(filtered_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 1)
    cv2.putText(filtered_image, f'{label} {conf:.2f}', (int(xmin), int(ymin) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
filtered_results = []
for *box, conf, cls in results.xyxy[0]:
    label = model.names[int(cls)]
    if label in desired_classes and conf >= confidence_threshold:
        filtered_results.append((box, conf, cls))

# Преобразование изображения в массив NumPy
image_np = np.array(image)

# Вырезание и отображение отфильтрованных объектов
cropped_images = []
for box, conf, cls in filtered_results:
    xmin, ymin, xmax, ymax = map(int, box)
    cropped_image = image_np[ymin:ymax, xmin:xmax]
    cropped_images.append(cropped_image)
    cv2.imshow("image", cropped_image)