import tensorflow as tf 

# model = tf.keras.models.load_model("./models/classification/google-vit-base-patch16-224")

converter = tf.lite.TFLiteConverter.from_saved_model("./models/classification/google-vit-base-patch16-224")

"""
Since subclassed models do not have a static input shape, 
you need to specify the input shape when converting to TFLite. 
You can use the tf.lite.TFLiteConverter class's 
experimental_new_converter parameter to set the input shape."""

input_shape = (1, 3, 224, 224)  
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Apply optimizations if desired
"""
This block is needed because Conv2D, a TF op, 
is not supported by the native TFLite runtime, 
hence you need to enable TF kernels fallback
"""
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
converter.allow_custom_ops = True  # If you have custom ops in the model
converter.input_shapes = {'input_name': input_shape}

tflite_model = converter.convert()
with open('./models/classification/google-vit-base-patch16-224.tflite', 'wb') as f:
  f.write(tflite_model)
