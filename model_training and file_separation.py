from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
import os, shutil
import cv2

train_data = '/content/drive/MyDrive/Driver Drowsiness Detection System/Train'
test_data = '/content/drive/MyDrive/Driver Drowsiness Detection System/Test'

for image in os.listdir(train_data):
          if image.split('_')[4] == '0':
            shutil.copy(src = '/content/drive/MyDrive/Driver Drowsiness Detection System/Train'+'/' + image, dst='/content/drive/MyDrive/Driver Drowsiness Detection System/Train/Closed Eyes' + '/' + image)
          if image.split('_')[4] == '1':
            shutil.copy(src = '/content/drive/MyDrive/Driver Drowsiness Detection System/Train' + '/' + image, dst = '/content/drive/MyDrive/Driver Drowsiness Detection System/Train/Open Eyes' + '/' + image)


for image in os.listdir(test_data):
  if image.split('_')[4] == '0':
    shutil.copy(src = '/content/drive/MyDrive/Driver Drowsiness Detection System/Test' + '/' + image, dst = '/content/drive/MyDrive/Driver Drowsiness Detection System/Test/Closed Eyes' + '/' + image)
  if image.split('_')[4] == '1':
    shutil.copy(src = '/content/drive/MyDrive/Driver Drowsiness Detection System/Test' + '/' + image, dst = '/content/drive/MyDrive/Driver Drowsiness Detection System/Test/Open Eyes' + '/' + image)

i = 0
for imge in os.listdir(test_data + '/Open Eyes'):
  i = i+1
print(i)

train_datagen = ImageDataGenerator(
                      rotation_range =0.2,
                      width_shift_range = 0.2,
                      height_shift_range = 0.2,
                      zoom_range = 0.2,
                      validation_split = 0.2,
                      rescale = 1./255)




train_dataset = train_datagen.flow_from_directory('/content/drive/MyDrive/Driver Drowsiness Detection System/Train',
                                              batch_size = 32, class_mode = 'categorical', target_size = (80, 80), subset = 'training')





val_dataset = train_datagen.flow_from_directory('/content/drive/MyDrive/Driver Drowsiness Detection System/Train',
                                         batch_size = 32, class_mode = 'categorical', target_size = (80, 80), subset = 'validation')




test_datagen = ImageDataGenerator(
                      rescale = 1./255,
                      validation_split = 0.2)

test_dataset = test_datagen.flow_from_directory('/content/drive/MyDrive/Driver Drowsiness Detection System/Test',
                                             batch_size = 32, class_mode = 'categorical', target_size = (80, 80) )



from tensorflow.keras.layers import Flatten, Dense, Dropout

bmodel = InceptionV3(include_top = False, weights = 'imagenet',  input_shape = (80, 80, 3))
hmodel = bmodel.output
hmodel = Flatten()(hmodel)
hmodel = Dense(64, activation = 'relu')(hmodel)
hmodel = Dropout(0.5)(hmodel)
hmodel = Dense(2, activation = 'softmax')(hmodel)

from tensorflow.keras.models import Model

model = Model(inputs = bmodel.input, outputs = hmodel)
for layer in bmodel.layers:
  layer.trainable = False


model.summary()