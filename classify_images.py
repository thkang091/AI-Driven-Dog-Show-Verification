from classifier import classifier
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16

def classify_images(images_dir, results_dic, model_name):
    model = load_model(model_name)
    for key in results_dic:
        img_path = f"{images_dir}/{key}"
        model_label = classifier(img_path, model)
        model_label = model_label.lower().strip()
        truth = results_dic[key][0]
        if truth in model_label:
            results_dic[key].extend([model_label, 1])
        else:
            results_dic[key].extend([model_label, 0])
    
    print("Classification Results:")
    for key, value in results_dic.items():
        print(f"Image: {key}, Label: {value[0]}, Predicted: {value[1]}, Match: {value[2]}")

def load_model(model_name):
    if model_name == "resnet":
        model = ResNet50(weights='imagenet')
        model.name = "resnet"
    elif model_name == "vgg":
        model = VGG16(weights='imagenet')
        model.name = "vgg"
    else:
        raise ValueError("Model name must be 'resnet' or 'vgg'.")
    return model
