import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess_input, decode_predictions as resnet_decode_predictions
from keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess_input, decode_predictions as vgg_decode_predictions
import numpy as np
import os

def classifier(img_path, model):
    if not os.path.exists(img_path):
        raise ValueError("Image path is not valid or image file does not exist.")
    
    # Determine preprocessing and decode functions based on the model type
    model_name = model.name
    if "resnet" in model_name:
        preprocess_input = resnet_preprocess_input
        decode_predictions = resnet_decode_predictions
    elif "vgg" in model_name:
        preprocess_input = vgg_preprocess_input
        decode_predictions = vgg_decode_predictions
    else:
        raise ValueError("Unsupported model type.")

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Classify the image
    preds = model.predict(x)
    label = decode_predictions(preds, top=1)[0][0][1]  # Get the predicted label
    
    return label
