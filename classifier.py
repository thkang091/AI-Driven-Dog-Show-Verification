import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import ast
import os

# Load pretrained VGG16 model
vgg_model = VGG16(weights='imagenet')

# Obtain ImageNet labels
with open('imagenet1000_clsid_to_human.txt') as imagenet_classes_file:
    imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())

def classifier(img_path, model_name='vgg', custom_model_path=None):
    if model_name == 'vgg':
        model = vgg_model
    else:
        if custom_model_path and os.path.exists(custom_model_path):
            model = load_model(custom_model_path)
        else:
            raise ValueError("Custom model path is not valid or model file does not exist.")
    
    # Load the image
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Make predictions
    predictions = model.predict(img_array)
    
    # Decode the predictions
    if model_name == 'vgg':
        decoded_predictions = decode_predictions(predictions, top=1)[0]
        pred_class = decoded_predictions[0][1]
        pred_idx = np.argmax(predictions[0])
        human_readable_class = imagenet_classes_dict.get(str(pred_idx), pred_class)
    else:
        pred_idx = np.argmax(predictions[0])
        human_readable_class = imagenet_classes_dict.get(str(pred_idx), "Unknown")
    
    return human_readable_class

def train_custom_model(train_data_dir, custom_model_path, epochs=10, batch_size=20):
    # Define the custom model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(120, activation='softmax')  # Assuming 120 dog breeds
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Data preparation
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Train the model
    model.fit(train_generator, epochs=epochs, steps_per_epoch=len(train_generator))

    # Save the model
    model.save(custom_model_path)

    return model
