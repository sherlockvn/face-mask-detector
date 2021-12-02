from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.applications.mobilenet import MobileNet
import keras.optimizer_v2.adam as adam
from keras.models import Model

from config import *

def create_base_model():
  #use light weight deep neutral networks 
  base_model = MobileNet(
      weights= None, 
      include_top=False, 
      input_shape= (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
  )
  # get one output tensor of the layer
  x = base_model.output
  # apply global average pooling operation to output tensor to reduce the dimension of data
  x = GlobalAveragePooling2D()(x)
  # add fully connected layer with 256 units and the rectified linear unit(relu) activation function.
  x = Dense(256,activation='relu')(x) 
  # add dropout layer which set input units to 0 with a frequency of rate = 0.2 for avoiding overfitting
  x = Dropout(0.2)(x)
  # last layer is fully connected layer with number of unit equals to number of class, use softmax activation
  # because softmax output is probability distribution of vector inputs
  predictions = Dense(NUM_CLASSES, activation='softmax')(x)
  model = Model(inputs=base_model.input, outputs=predictions)
  # use Adam optimizer with learing rate is 0.000125
  opt = adam.Adam(lr=0.000125)
  # configure model with loss function is categorial cross entropy because of multi-class classfication task
  # and metric is accuracy only before start training 
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model
