import cv2 
from PIL import Image 
import numpy as np 
import glob
import random 
import torch 
from typing import Optional, Any 
from torch.utils.mobile_optimizer import optimize_for_mobile

# MODELS ROOT PATH
# /Users/marvinmboya/.cache/torch/hub/checkpoints/

SEED = 42
np.random.seed(SEED)
# random.seed(SEED)

# Get the path to the directory
sample_images_dir = "./imagenet-sample-images/"
sample_images: Optional[list[str]] = None

def load_image(image_path: str) -> np.ndarray:
    image_bgr = cv2.imread(image_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image

def show_image(
    image: np.ndarray, 
    prediction: Optional[str] = None
    ) -> None:  
    if prediction:
        image = cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow("image", RGB2BGR(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cv2_to_pil(input_image: np.ndarray) -> Image.Image:
    output_image = Image.fromarray(input_image)
    return output_image

def pil_to_cv2(input_image: Image.Image) -> np.ndarray:
    output_image = np.asarray(input_image)
    return output_image

def RGB2BGR(input_image: np.ndarray) -> np.ndarray:
    output_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    return output_image

def sample_images(dir: str = sample_images_dir) -> list[str]:
    image_files = glob.glob(dir + "/*.JPEG")
    return image_files


def post_quantization(
    model: torch.nn.Module, 
    backend: str = "qnnpack") -> torch.nn.Module:
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8)
    quantized_model = torch.jit.script(model_int8)
    return quantized_model

def jit_script(model: torch.nn.Module) -> torch.jit.ScriptModule:
    scripted: torch.jit.ScriptModule = torch.jit.script(model)
    return scripted

def optimize_trace(
    model: Any, 
    input_shape: list[int],
    dir: str ="classification", 
    name="model") -> None:
    example = torch.rand(*input_shape)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter(
        f"./models/{dir}/{name}.ptl"
        )

def model_save(
    model: Any, 
    dir: str ="classification", 
    mode: str = "tf",
    name="model"
    ):
    if mode == "tf":
        model.save(f'./models/{dir}/{name}.h5')
    elif mode == "script":
        model.save(f"./models/{dir}/{name}.pt")
    elif mode == "pt":
        torch.save(model, f"./models/{dir}/{name}.pt")

def sample_image() -> Optional[np.ndarray]:
    if not sample_images:
        print("Sample images not loaded")
        return None
    sample_image_path: str = random.choice(sample_images)
    image: np.ndarray = load_image(sample_image_path)
    return image

sample_images = sample_images()




