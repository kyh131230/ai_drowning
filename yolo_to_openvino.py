from ultralytics import YOLO

model = YOLO('yolo26l-pose.pt')

model.export(format='openvino', dynamic=True, imgsz=1280)
# model.export(format='openvino')