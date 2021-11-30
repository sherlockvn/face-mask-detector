from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.applications.mobilenet import MobileNet
import keras.optimizer_v2.adam as adam
from keras.models import Model

from config import *

def create_base_model():
  #base model
  base_model = MobileNet(
      weights= None, 
      include_top=False, 
      input_shape= (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
  )

  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(256,activation='relu')(x) 
  x = Dropout(0.2)(x)
  predictions = Dense(NUM_CLASSES, activation='softmax')(x)
  model = Model(inputs=base_model.input, outputs=predictions)

  opt = adam(lr=0.000125)
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model
