from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.xception     import Xception
from keras.applications.resnet       import ResNet50
from keras.applications.resnet_v2    import ResNet50V2
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.layers import Input
from keras.optimizers import SGD

import numpy as np
import cv2

# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(64, 64, 3))

# create the base pre-trained model
base_model = MobileNetV2(input_tensor=input_tensor, weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
print(x)
# x = GlobalAveragePooling2D()(x)
# # let's add a fully-connected layer
x = Dense(512, activation='relu')(x)
# # and a logistic layer -- let's say we have 200 classes
predictions = Dense(200, activation='softmax')(x)

# # this is the model we will train
model = Model(input_tensor, outputs=predictions)

# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# # # we train our model again (this time fine-tuning the top 2 inception blocks
# # # alongside the top Dense layers
# model.fit_generator(...)
