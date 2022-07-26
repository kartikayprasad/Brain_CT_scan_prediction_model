# A python file compatible with python version 3 and above.

# Importing the Library

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks

# Defining the data path
training_data = 'E:/Udemy/Brain_tumor/BrainTD-master/data/brain_tumor_dataset'

train_datagen = ImageDataGenerator(rescale= 1./255, height_shift_range=0.2, width_shift_range=0.2, zoom_range=0.2,
                                   validation_split = 0.2, fill_mode = 'nearest', shear_range=0.2, horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(training_data, target_size = (200,200), batch_size = 10, class_mode = 'categorical', subset='training')
validation_generator = train_datagen.flow_from_directory(training_data, target_size = (200,200), batch_size = 10, class_mode = 'categorical', subset='validation')


# model designing

brain_scan = tf.keras.models.Sequential()
brain_scan.add(Conv2D(16, kernel_size= 3, activation= 'relu', input_shape= [200,200,3]))
brain_scan.add(MaxPool2D(pool_size=2,strides=2,padding='same'))
brain_scan.add(Conv2D(32, kernel_size= 3, activation= 'relu'))
brain_scan.add(MaxPool2D(pool_size=2,strides=2,padding='same'))
brain_scan.add(Conv2D(64, kernel_size= 3, activation= 'relu'))
brain_scan.add(MaxPool2D(pool_size=2,strides=2,padding='same'))
brain_scan.add(Conv2D(64, kernel_size= 3, activation= 'relu'))
brain_scan.add(MaxPool2D(pool_size=2,strides=2,padding='same'))
brain_scan.add(Conv2D(64, kernel_size= 3, activation= 'relu'))
brain_scan.add(MaxPool2D(pool_size=2,strides=2,padding='same'))
brain_scan.add(Conv2D(128, kernel_size= 3, activation= 'relu'))
brain_scan.add(MaxPool2D(pool_size=2,strides=2,padding='same'))
brain_scan.add(GlobalAveragePooling2D(name='avgpool'))
# brain_scan.add(Dense(units = 128, activation='relu'))
brain_scan.add(Dense(units = 2, activation='softmax'))

#setting up the callbacks
my_callback = [callbacks.EarlyStopping(monitor= 'val_loss', patience=10),
               callbacks.ModelCheckpoint(filepath='brain_scan.{epoch:02d}-{val_loss:.2f}.h5', save_best_only=True, save_weights_only=False)]

#Model compiling
brain_scan.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics=['accuracy'])

brain_scan.summary()

# Fitting the model
history = brain_scan.fit(
     train_generator,
     epochs=200,
     validation_data=validation_generator,verbose=1,callbacks= my_callback)

print("Model Loaded")
print(history)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

from tensorflow.keras.models import load_model
brain_model = load_model("E:/Pyhton_learning/firstprog/brain_scan.23-0.43.h5")

import os
from tensorflow.keras.preprocessing import image
list = sorted(os.listdir('E:/Udemy/Brain_tumor/BrainTD-master/data/Prediction'))

print(train_generator.class_indices)

for label in list:
    path = "E:/Udemy/Brain_tumor/BrainTD-master/data/Prediction"
    file = path + '/' + label
    test_image = image.load_img(file, target_size = (200,200))
    test_image = image.img_to_array(test_image)
    # test_image = preprocess_input(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = brain_model.predict(test_image)


    if result[0][0] > result[0][1]:
        result1 = 'No_tumours'
    else:
        result1 = 'Tumours'

    print(f"{label} is {result1} and the probabilty for both the No_tumour and Tumor is: {result}")

