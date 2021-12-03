from keras.layers import AveragePooling2D, Dropout, Dense
from keras.applications.mobilenet import MobileNet
from keras.layers.core.flatten import Flatten
import keras.optimizer_v2.adam as adam
from keras.models import Model

from config import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS,NUM_CLASSES

def create_base_model():
  #use light weight deep neutral networks 
  base_model = MobileNet(
      weights= "imagenet", 
      include_top=False, 
      input_shape= (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
  )
  # get one output tensor of the layer
  x = base_model.output
  # apply average pooling operation to output tensor to reduce the dimension of data
  # because average pooling seem to retains much information about the less important elements of block(pool)
  x = AveragePooling2D(pool_size=(6, 6))(x)
  # flatten to 1D array before add dense layer
  x = Flatten(name="flatten")(x)
  # add fully connected layer with 256 units and the rectified linear unit(relu) activation function.
  x = Dense(256,activation='relu')(x) 
  # add dropout layer which set input units to 0 with a frequency of rate = 0.4 for avoiding overfitting
  x = Dropout(0.4)(x)
  # last layer is fully connected layer with number of unit equals to number of class, use softmax activation
  # because softmax output is probability distribution of vector inputs
  predictions = Dense(NUM_CLASSES, activation='softmax')(x)
  model = Model(inputs=base_model.input, outputs=predictions)
  # use Adam optimizer with learing rate is 0.000125
  opt = adam.Adam(lr=0.000125)
  # configure model with loss function is categorial cross entropy because of multi-class classification task
  # and metric is accuracy only before start training 
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model
