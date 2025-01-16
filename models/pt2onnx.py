#!/usr/bin/python3

from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')

model.export(format="onnx")

onnx_model = YOLO("yolo11n-pose.onnx")