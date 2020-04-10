from keras.preprocessing import image
from keras.models import load_model
import numpy as np

img_width, img_height = 224, 224

if __name__ == '__main__':
    model = load_model('model_saved.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    img = image.load_img('cucumber4.png', target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)
    outcome = classes[0][0]
    if(outcome == 1): print("zucchini")
    else: print("cucumber")
