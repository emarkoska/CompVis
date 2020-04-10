from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers
import os, os.path
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from keras.callbacks import EarlyStopping

img_width, img_height = 224, 224


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


#Assemble the data
train_data_dir = 'Data/all'
validation_data_dir = 'Data/test'


nb_train_samples = len(os.listdir("Data/all/cucumber")) + len(os.listdir("Data/all/zucchini"))
nb_validation_samples = len(os.listdir("Data/test/cucumber")) + len(os.listdir("Data/test/zucchini"))

#Resize the images

#Train
epochs = 10
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#es = EarlyStopping(monitor='val_loss', mode='min', patience=7, verbose=1)
optimiser = optimizers.adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss='binary_crossentropy',
              optimizer=optimiser,
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='binary')


# history = model.fit(train_generator,
#                     steps_per_epoch=nb_train_samples // batch_size,
#                     epochs=epochs, validation_data=validation_generator,
#                     validation_steps=nb_validation_samples // batch_size)

history = model.fit(train_generator,
                     steps_per_epoch=nb_train_samples // batch_size,
                     epochs=epochs)

#plot_history(history)

model.save('model_saved.h5')

