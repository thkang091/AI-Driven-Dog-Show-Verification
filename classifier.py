import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np
import ast

# Load pretrained model
model = VGG16(weights='imagenet')

# Obtain ImageNet labels
with open('imagenet1000_clsid_to_human.txt') as imagenet_classes_file:
    imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())

def classifier(img_path, model_name='vgg'):
    # Load the image
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Make predictions
    predictions = model.predict(img_array)
    
    # Decode the predictions
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    pred_class = decoded_predictions[0][1]
    
    pred_idx = np.argmax(predictions[0])
    human_readable_class = imagenet_classes_dict.get(str(pred_idx), pred_class)
    
    return human_readable_class
