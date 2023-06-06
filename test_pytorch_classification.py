from PIL import Image
import torch
from torchvision.models import (
    mobilenet_v3_large, 
    MobileNet_V3_Large_Weights,
    MaxVit_T_Weights,
    maxvit_t
)

from utils_core import (
    sample_image, 
    show_image, 
    cv2_to_pil, 
    post_quantization,
    jit_script,
    model_save
)
from utils_pre_post_inference import pt_class_preprocess_inference

from typing import Any 
from classes import ImageNet_Classes

classes = ImageNet_Classes().classes
np_image: Any = sample_image()
image = cv2_to_pil(np_image)

weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
model = mobilenet_v3_large(weights=weights)
model.eval()

"""
weights = MaxVit_T_Weights.IMAGENET1K_V1
model = maxvit_t(weights=weights)
model.eval()
"""

category_name, score = pt_class_preprocess_inference(image, weights, model)
print(f"{category_name}: {100 * score:.1f}%")
# show_image(np_image, f"{category_name}: {100 * score:.1f}%")

quantized_model = post_quantization(model)
scripted_model = jit_script(quantized_model)

# model_save(scripted_model, mode="script", name="mobilenet-v3-imagenet1k-v2")

category_name, score = pt_class_preprocess_inference(image, weights, scripted_model)
print(f"{category_name}: {100 * score:.1f}%")
# show_image(np_image, f"{category_name}: {100 * score:.1f}%")
