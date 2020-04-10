from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import sys
import tqdm
import json

model = ResNet50(weights='imagenet')

img = image.load_img(sys.argv[1],target_size=(224,224))
x=image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)
pred = np.argmax(model.predict(x))
if ((pred <= 268) & (pred >= 151)): print("dog")
if ((pred <= 285) & (pred >= 281)): print("cat")
else: print("Unknown")


