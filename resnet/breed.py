from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import sys
import json

model = ResNet50(weights='imagenet')
img = image.load_img(sys.argv[1], target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
pred = decode_predictions(preds, top = 5)[0]
dict = dict()

for item in pred:
    dict[item[1]] = str(item[2])

pred_json = json.dumps(dict)
print('Predicted:', pred_json)

with open("%s-pred.json" % sys.argv[1].split("/")[-1].split(".")[-2], 'w') as f:
    f.write(pred_json)


