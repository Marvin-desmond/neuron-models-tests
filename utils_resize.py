import cv2 
import numpy as np
import torch 
import tensorflow as tf

from typing import Union 

def resize_image(image, size):
    size = (size, size) if isinstance(size, int) else size
    image = cv2.resize(image, size)
    return image

def xyxy_to_xywh(bounding_boxes):
    xmin, ymin, xmax, ymax = bounding_boxes
    width = xmax - xmin
    height = ymax - ymin
    xcenter = xmin + int(width / 2)
    ycenter = ymin + int(height / 2)
    return xcenter, ycenter, width, height

def xywh_to_xyxy(bounding_boxes):
    x_center, y_center, width, height = bounding_boxes
    return (
        x_center - int(width / 2),
        y_center - int(height / 2), 
        x_center + int(width / 2), 
        y_center + int(height / 2)
    )

def batch_xywh_to_xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def resize_bounding_boxes( old_size, bounding_boxes, new_size):
    height, width = old_size
    height_ratio, width_ratio = new_size[0] / height, new_size[1] / width
    x, y, w, h = bounding_boxes
    x, w = int(width_ratio * x), int(width_ratio * w)
    y, h = int(height_ratio * y), int(height_ratio * h)
    return x, y, w, h