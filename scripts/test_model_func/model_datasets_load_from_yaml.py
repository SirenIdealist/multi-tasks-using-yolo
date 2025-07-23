"""
This script is used to test the functionality of loading model datasets from a YAML configuration file.
"""

from ultralytics import YOLO

model = YOLO("/home/ubuntu/repos/multi-tasks-using-yolo/ultralytics/cfg/models/tomato_peduncle/tomato_peduncle_multi_tasks_model_model.yaml",
             task="multi_tasks")

model.train(data="/home/ubuntu/repos/multi-tasks-using-yolo/ultralytics/cfg/datasets/tomato_peduncle/tomato_peduncle_multi_tasks_model_datasets.yaml",
            task="multi_tasks",
            batch=16,
            epochs=10,
            imgsz=640,
            device="0")