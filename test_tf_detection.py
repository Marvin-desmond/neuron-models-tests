import tensorflow as tf
import numpy as np 
import random 
from pprint import pprint
random.seed(42)

from utils_core import load_image, show_image
from utils_resize import resize_image

from utils_pre_post_inference import (
    tf_preprocess,
    tf_inference,  
    tf_non_max_suppression,
    tf_draw_predictions,
)

inf_size = 640
classes = open("coco.names").read().strip().split('\n')
colors = [[random.randint(0, 255) for _ in range(3)] for _ in classes]


original_image: np.ndarray = load_image("./people.png")
original_size = original_image.shape[:2]
inference_image = resize_image(original_image, inf_size)

Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate

# ENSURE YOU HAVE SAVED THE MODEL BELOW IN YOUR LOCAL FOLDER
interpreter = Interpreter(model_path="./yolov5m-fp16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
norm_image = tf_preprocess(inference_image)
interpreter.set_tensor(input_details['index'], norm_image)
interpreter.invoke()
output_tensor = interpreter.get_tensor(output_details['index'])

b, h, w, ch = norm_image.shape

y = interpreter.get_tensor(output_details['index'])
y = y if isinstance(y, np.ndarray) else y.numpy()
y[..., :4] *= [w, h, w, h]
outputs =  tf.convert_to_tensor(y)
res = tf_non_max_suppression(outputs)
drawer_image = tf_draw_predictions(np.copy(original_image), res[0], classes, colors, [inf_size, inf_size], original_image.shape[:2])
show_image(drawer_image, "Tensorflow")
