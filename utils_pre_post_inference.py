import numpy as np 
import tensorflow as tf

import torch 
import torch.nn as nn 
import torchvision 
import cv2 

import time 
import random 

from utils_resize import resize_bounding_boxes, resize_image, xywh_to_xyxy, xyxy_to_xywh, batch_xywh_to_xyxy

class Ensemble(nn.ModuleList):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = [module(x)[0] for module in self]
        y = torch.cat(y, 1) 
        return y, None

def init_model(variant, device):
    model = Ensemble()
    ckpt = torch.load(f"./sample-ultralytics/{variant}", map_location='cpu') 
    ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float() 
    model.append(ckpt.fuse().eval()) 
    model = model[-1]
    model.float()

    for m in model.modules():
        if isinstance(m, nn.Upsample):
            m.recompute_scale_factor = None
    # model metadata
    classes = open("coco.names").read().strip().split('\n')
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in classes]
    return model, colors, classes

def pt_class_preprocess_inference(img, weights, model):
    preprocess = weights.transforms()
    batch = preprocess(img).unsqueeze(0)
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    return (category_name, score)


def pt_preprocess(input_img, model, device):
    img = torch.from_numpy(input_img).to(device)
    img = img.float() # UINT8 -> FLOAT32 
    img /= 255.0 # [0, 255] -> [0, 1]
    img = img.unsqueeze(0) # [HWC] -> [BHWC] 
    img = img.permute(0, 3, 1, 2) # BHWC -> BCHW
    return img

def tf_preprocess(input_img):
    image = tf.convert_to_tensor(input_img)
    image = tf.cast(image, tf.float32)
    image = image / 255.0 
    image = tf.expand_dims(image, axis = 0)
    # image = tf.transpose(image, [0, 3, 1, 2])
    return image

def pt_inference(model, image):
    prediction = model(image)
    prediction = prediction[0]
    return prediction.detach()

def tf_inference(model, image):
    prediction = model(image)
    prediction = prediction[0]
    return prediction

def pt_classification(prediction, labels):
    index = prediction.argmax().item()
    score = prediction[index].item()
    category_name = labels[index]
    return category_name, score

def box_iou(box1, box2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (np.min(box1[:, None, 2:], box2[:, 2:]) - np.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)



def numpy_native_non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    _, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into np.argsort()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
    then = time.time()
    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        print("SHAPE ==> ", x.shape, xi)
        x = x[xc[xi]]  # confidence
        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = np.zeros((len(l), nc + 5))
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[np.arange(len(l)), l[:, 0].astype(int) + 5] = 1.0  # cls
            x = np.concatenate((x, v), 0)
        # If none remain process next image
        if x.shape[0] == 0:
            continue
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        box = batch_xywh_to_xyxy(x[:, :4])
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            indices = np.where(x[:, 5:] > conf_thres)
            i, j = indices[0], indices[1]
            x = np.concatenate((box[i], np.expand_dims(x[i, j + 5], 1), np.expand_dims(j.astype(float), 1)), 1)
        else:  # best class only
            conf = np.max(x[:, 5:], axis=1, keepdims=True)
            j = np.argmax(x[:, 5:], axis=1)
            j = np.expand_dims(j, axis=1)
            x = np.concatenate((box, conf, j.astype(float)), 1)
            x = x[conf.squeeze() > conf_thres]

        # Filter by class
        if classes is not None:
            class_tensor = np.array(classes)
            x = x[np.any(x[:, 5:6] == class_tensor, axis=1)]
        n = x.shape[0]  # number of boxes
        if n == 0:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            sorted_indices = np.argsort(x[:, 4])[::-1]
            x = x[sorted_indices[:max_nms]]  # sort by confidence
        # Batched NMS
        boxes, scores = x[:, :4], x[:, 4]  # boxes, scores
        i = tf.image.non_max_suppression(boxes, scores, 100, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[:, None]  # box weights
            x[i, :4] = np.matmul(weights, x[:, :4]) / np.sum(weights, axis=1, keepdims=True)  # merged boxes
            if redundant:
                i = i[np.sum(iou, axis=1) > 1]  # require redundancy
        output[xi] = x[i]
        now = time.time()
        if (now - then) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
    return output


def tf_box_iou(box1, box2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    area1 = box_area(tf.transpose(box1))
    area2 = box_area(tf.transpose(box2))
    inter = (np.min(box1[:, None, 2:], box2[:, 2:]) - np.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def tf_non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    nc = prediction.shape[2] - 5  # number of classes
    above_thres = tf.math.greater(prediction[..., 4], conf_thres)  # candidates
    max_det = 300  # maximum number of detections per image
    output = [tf.zeros((0, 6))] * prediction.shape[0]
    x = prediction[0]
    x = tf.boolean_mask(x, above_thres[0])
    obj_cls_conf = x[:, 5:] * x[:, 4:5]
    x = tf.concat([x[:, :5], obj_cls_conf], 1)
    box = batch_xywh_to_xyxy(x[:, :4])
    conf = tf.math.reduce_max(x[:, 5:], axis=1, keepdims=True)
    j = tf.argmax(x[:, 5:], 1)
    j = tf.expand_dims(j, 1)
    x = tf.concat([box, conf, tf.cast(j, tf.float32)], 1)
    x = x[tf.squeeze(conf) > conf_thres]
    n = x.shape[0]
    boxes, scores = x[:, :4], x[:, 4]  # boxes, scores
    i = tf.image.non_max_suppression(boxes, scores, 100, iou_thres)  # NMS
    if i.shape[0] > max_det:  # limit detections
        i = i[:max_det]
    output[0] = tf.gather(x, i)
    return output

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    prediction = prediction.to("cpu")
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    _, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
    then = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)
        # If none remain process next image
        if not x.shape[0]:
            continue
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        box = batch_xywh_to_xyxy(x[:, :4])
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        output[xi] = x[i]
        now = time.time()
        if (now - then) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
    return output

def draw_predictions(image, predictions, classes, colors, inference_tuple, final_tuple):
    for prediction in predictions:
        cls = prediction[-1]
        box = prediction[:4]
        box = box.int().numpy()
        box = xywh_to_xyxy(resize_bounding_boxes(inference_tuple, xyxy_to_xywh(box), final_tuple))
        image = cv2.resize(image, [final_tuple[1], final_tuple[0]])
        cv2.rectangle(image, 
            (box[0], box[1]), 
            (box[2], box[3]), 
            color=colors[int(cls)], 
            thickness=2)
        cv2.putText(image, 
        f"{classes[int(cls)]}", 
        (box[:2][0], box[:2][1]), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, (0, 0, 255), 2)
    return image

def tf_draw_predictions(image, predictions, classes, colors, inference_tuple, final_tuple):
    for prediction in predictions:
        cls = prediction[-1]
        box = prediction[:4]
        box = box.numpy().astype(int)
        box = xywh_to_xyxy(resize_bounding_boxes(inference_tuple, xyxy_to_xywh(box), final_tuple))
        image = cv2.resize(image, [final_tuple[1], final_tuple[0]])
        cv2.rectangle(image, 
            (box[0], box[1]), 
            (box[2], box[3]), 
            color=colors[int(cls)], 
            thickness=2)
        cv2.putText(image, 
        f"{classes[int(cls)]}", 
        (box[:2][0], box[:2][1]), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, (0, 0, 255), 2)
    return image