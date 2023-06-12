import tensorflow as tf 
from transformers import AutoImageProcessor, TFViTForImageClassification
from datasets import load_dataset
from pprint import pprint 
from utils_core import load_image, cv2_to_pil

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

np_image = load_image("./cat.png")
image = cv2_to_pil(np_image)

model = TFViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

"""
CUSTOM INFERENCE
"""

custom_preprocessor = tf.keras.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 0)), # H,W,C -> 1, H, W, C
  tf.keras.layers.Resizing(224, 224), # scale(H), scale(W)
  tf.keras.layers.Rescaling(1.0 / 255), # H, W in [0, 1]
  tf.keras.layers.Normalization(axis=-1, mean=(.5, .5, .5), variance=[i ** 2 for i in (.5, .5, .5)]), # (input - mean) / sqrt(var)
  tf.keras.layers.Lambda(lambda x: tf.transpose(x, [0, 3, 1, 2])) # 1, H, W, C -> 1, C, H, W
])

custom_inputs = custom_preprocessor(np_image)
custom_logits = model(custom_inputs).logits
predicted_label = int(tf.math.argmax(custom_logits, axis=-1))

print(model.config.id2label[predicted_label])
"""
HF INFERENCE
"""

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
inputs = image_processor(image, return_tensors="tf") # 1, 3, 224, 224

logits = model(**inputs).logits

predicted_label = int(tf.math.argmax(logits, axis=-1))
print(model.config.id2label[predicted_label])
model.save("./models/classification/google-vit-base-patch16-224")