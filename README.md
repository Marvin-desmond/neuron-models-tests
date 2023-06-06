
  <div style="display: flex; justify-content: center; align-items: center; width: 100%; border: 1px solid red;">
    <img alt="Neuron" src="./logo.png" width="52" height="42">
    <span style="color: #dadada; font-weight: bold; font-size: 35px; margin: 0 3%;">Neuron</span>
  </div>
  <br/>
  <br/>
<h4 align="center">
    <p>
       | <b>English</b> |
    <p>
</h4>

<h3 align="center">
    <p>One-for-All Flutter package for TensorFlow and PyTorch for inference of TFLlite and TorchScript models</p>
</h3>


<img src="./logo.png" width="20" style="margin-right: 5px;"/> Neuron provides a Flutter tool for integration with the Android native tools for TensorFlow and PyTorch for preprocessing images and performing inference across classification, segmentation and detection models to perform tasks on different modalities such as text, vision, and audio.
For now, I have started with vision, later I will update the repo with audio and text.


### May 21, 2023
* Preprocesing column implies whether preprocessing of image inputs is done before or inside the model. It is always recommended that when implementing a model solution, preprocessing to be part of the model.

| model                                                                                                                |task  | framework |file_type | normalization | preprocessing | image size | model size | tested |
|----------------------------------------------------------------------------------------------------------------------|------|------|--------|-----------|------|-----|-------|-------|
| [lite-model_imagenet_mobilenet_v3_large_100_224_classification_5_default_1](https://tfhub.dev/google/lite-model/imagenet/mobilenet_v3_large_100_224/classification/5/default/1) | classification | <img src="./tensorflow.png" width="20"> | tflite | [0, 1] |before | 224 | 21MB | :white_check_mark: |
| [lite-model_imagenet_mobilenet_v3_small_075_224_classification_5_default_1](https://tfhub.dev/google/lite-model/imagenet/mobilenet_v3_small_075_224/classification/5/default/1)                                                               |classification | <img src="./tensorflow.png" width="20"> | tflite | [0, 1] | before | 224 | 7.8MB | :white_check_mark: |
|[multi-axis-vision-transformer-maxvit-t](https://pytorch.org/vision/stable/models/generated/torchvision.models.maxvit_t.html#torchvision.models.MaxVit_T_Weights) | classification | <img src="./pytorch.png" width="20"> | pytorch | [0, 1]<br>Normalization<br>mean=[0.485, 0.456, 0.406]<br>std=[0.229, 0.224, 0.225] | before | 224 | 55MB(quantized, scripted) |:x:|
| [model.pt](https://github.com/pytorch/android-demo-app/blob/master/HelloWorldApp/app/src/main/assets/model.pt) | classification | <img src="./pytorch.png" width="20"> | pytorch | [0, 1]<br>Normalization<br>mean=[0.485, 0.456, 0.406]<br>std=[0.229, 0.224, 0.225] | before | 224 | 19.4MB | :white_check_mark: |
| [mobilenet-v3-imagenet1k-v2](https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html#torchvision.models.MobileNet_V3_Large_Weights) | classification |  <img src="./pytorch.png" width="20"> | pytorch | [0, 1]<br>Normalization<br>mean=[0.485, 0.456, 0.406]<br>std=[0.229, 0.224, 0.225] | before | 224 (central crop) | 14MB (quantized, scripted) | :x: |
| [yolov5m-fp16](https://github.com/Marvin-desmond/Spoon-Knife/releases/download/v1.0/yolov5m-fp16.tflite) | detection | <img src="./tensorflow.png" width="20"> | tflite | [0, 1] | before | 640 | 40.5MB | :white_check_mark: |
| [yolov5s](https://github.com/pytorch/android-demo-app/blob/master/ObjectDetection/README.md) | detection | <img src="./pytorch.png" width="20"> | pytorch | [0, 1] | before | 640 | 30MB | :white_check_mark: |


In general, these models, when finished, can be applied on offline apps that deal with:

* 🖼️ Images, for tasks like image classification, object detection, and segmentation.
* 📝 Text, for tasks like text classification, information extraction, question answering, summarization, translation, text generation, in over 100 languages.
* 🗣️ Audio, for tasks like speech recognition and audio classification.


