from ultralytics import YOLO

model = YOLO("runs/detect/train38/weights/best.pt")  # load a pretrained model (recommended for training)
success = model.export(format="onnx")  # export the model to ONNX format