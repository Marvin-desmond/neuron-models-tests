import numpy as np 
import torch 
import random

from utils_core import load_image, show_image
import sys
from utils_resize import resize_image
from utils_pre_post_inference import (
    init_model,
    pt_preprocess, 
    pt_inference, 
    non_max_suppression,
    draw_predictions
)
from test_download import attempt_download

inf_size = 640
model_name = "yolov5l.pt"
device = torch.device("cpu")

model_path, downloaded = attempt_download(model_name)

if downloaded:
    print(model_path)
else:
    print("Oooops!")
    sys.exit()

model, colors, classes = init_model(model_name, device)
"""
model = torch.jit.load("./sample-ultralytics/yolov5s.torchscript.ptl")
classes = open("coco.names").read().strip().split('\n')
colors = [[random.randint(0, 255) for _ in range(3)] for _ in classes]
"""
model.eval()

original_image: np.ndarray = load_image("./people.png")
original_size = original_image.shape[:2]
inference_image = resize_image(original_image, inf_size)

norm_image = pt_preprocess(inference_image, model, device)
predictions = pt_inference(model, norm_image)

class NMSLambda(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conf_thres=0.30 
        self.iou_thres=0.30
        self.classes=None
        self.agnostic=False
        self.multi_label=False
        self.labels=()

    def forward(self, prediction):
        prediction = (prediction[0]).detach()
        return non_max_suppression(
            prediction, conf_thres=0.30, iou_thres=0.30, 
            classes=None, agnostic=False, multi_label=False,
            labels=())

bundled_model = torch.nn.Sequential(
    model, 
    NMSLambda()
)
bundled_model.eval()
tuned_image = pt_preprocess(inference_image, bundled_model, device)
predictions = pt_inference(bundled_model, tuned_image)

drawn_image = draw_predictions(np.copy(original_image), predictions, classes, colors, [inf_size, inf_size], original_image.shape[:2])
show_image(drawn_image, "Pytorch")






