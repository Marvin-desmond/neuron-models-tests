import tensorflow as tf 
import numpy as np 
from classes import ImageNet_Classes
from utils_core import sample_image, show_image, cv2_to_pil
from typing import Any
from pprint import pprint 

model_path = "./test_models/classification/lite-model_imagenet_mobilenet_v3_small_075_224_classification_5_default_1.tflite"

image: Any = sample_image()
classes = ImageNet_Classes().classes

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.

tf_image = tf.keras.preprocessing.image.img_to_array(image)

norm_image = tf.image.resize(tf_image, [224, 224]) / 255.0
input_image = tf.expand_dims(norm_image, axis=0)
interpreter.set_tensor(input['index'], input_image)
interpreter.invoke()
output_tensor = interpreter.get_tensor(output['index'])

label = np.argmax(output_tensor[0])
probability = np.max(output_tensor[0])

show_image(image, classes[f"{label}"])
print(f"{classes[str(label)]}: {probability:.1f}%")

"""
sorted_indices = np.argsort(output_tensor[0])[::-1]
top_indices = sorted_indices[:5]
top_probabilities = output_tensor[0][top_indices]
"""

tf_image = tf.keras.preprocessing.image.img_to_array(image)

tf_image = tf.expand_dims(image, 0)
# norm_image = tf.image.resize(tf_image, [224, 224]) / 255.0
# norm_image = tf.keras.layers.Normalization(
#     axis=-1, 
#     mean=(.5, .5, .5), 
#     variance=[i ** 2 for i in (.5, .5, .5)])(norm_image)
# norm_image = tf.keras.layers.Lambda(lambda x: tf.transpose(x, [0, 3, 1, 2]))(norm_image)

interpreter = tf.lite.Interpreter(
    model_path = "./test_models/classification/lite-model_imagenet_mobilenet_v3_large_100_224_classification_5_default_1.tflite"
    )
interpreter.allocate_tensors()
output = interpreter.get_output_details()[0]
input = interpreter.get_input_details()[0] 

norm_image = tf.image.resize(tf_image, [224, 224]) / 255.0
input_image = tf.expand_dims(norm_image, axis=0)
interpreter.set_tensor(input['index'], norm_image)
interpreter.invoke()
output_tensor = interpreter.get_tensor(output['index'])

label = np.argmax(output_tensor[0])
probability = np.max(output_tensor[0])

show_image(image, classes[f"{label}"])
print(f"{classes[str(label)]}: {probability:.1f}%")

