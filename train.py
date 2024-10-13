import os

from ultralytics import YOLO


config_path = 'dataset.yaml'

model = YOLO("yolov5n.pt")

model.train(data=config_path, epochs=10, batch=32, workers=0)
model.tune()
