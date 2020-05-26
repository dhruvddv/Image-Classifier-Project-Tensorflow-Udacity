import tensorflow as tf
import tensorflow_hub as hub

import warnings
warnings.filterwarnings('ignore')
import logging 

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

from line_app import load_class_names, load_saved_model, process_image
from PIL import Image 
import numpy as np 
import argparse 



parser = argparse.ArgumentParser ()
parser.add_argument ('--image_path', default='./test_images/hard-leaved_pocket_orchid.jpg', help = 'Path to image file to make predictions on', type = str)
parser.add_argument('--model', help='Point to path of the saved Keras model', type=str)
parser.add_argument ('--top_k', default = 5, help = 'Top K most likely classes where K is an Integer', type = int)
parser.add_argument ('--class_label_map' , default = 'label_map.json', help = 'Mapping of categories to real names file, in JSON format', type = str)
commands = parser.parse_args()

print(commands)


image_path, model, K, class_label_map = commands.image_path, commands.model, commands.top_k, commands.class_label_map

def predict(image_path, model, K, cls_names):
    im = Image.open(image_path)
    prediction_image = np.asarray(im)
    prediction_image = process_image(prediction_image)
    
    prediction = model.predict(np.expand_dims(prediction_image, axis=0))

    top_values, top_idx = tf.math.top_k(prediction, K)
    
    top_classes = [class_names[str(value + 1)] for value in top_idx.numpy()[0]]
    
    return top_values.numpy()[0], top_classes


if __name__ == "__main__":
    
    project_ic_model = load_saved_model(model)
    class_names = load_class_names(class_label_map)
    
    print(predict(image_path,project_ic_model, K, class_names))