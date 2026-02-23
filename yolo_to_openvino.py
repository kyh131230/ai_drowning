from ultralytics import YOLO

model = YOLO('yolo26s.pt')

model.export(format='openvino', dynamic=True, imgsz=640)
# model.export(format='openvino')