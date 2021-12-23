# configs
from src.config.config import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS,NUM_CLASSES
# keras api
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Input
import keras.optimizer_v2.adam as adam
from keras.models import Model
from tensorflow.keras.applications import MobileNetV2

# create base model
def create_base_model():
  # use MobileNetV2 - a light weight deep neutral networks 
  base_model = MobileNetV2(
      weights= "imagenet", 
      include_top=False, 
      input_tensor=Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
  )

  # get one output tensor of the layer
  x = base_model.output

  # apply AveragePooling2D to output tensor to reduce the dimension of data
  # since AveragePooling2D will retain lots of information about the less important elements of block (pool)
  x = AveragePooling2D(pool_size=(6, 6))(x)

  # flatten to 1D array before add dense layer
  x = Flatten(name="flatten")(x)
  
  # add fully connected layer with 128 units and the rectified linear unit(relu) activation function.
  x = Dense(128,activation='relu')(x) 

  # add dropout layer which set input units to 0 with a frequency of rate = 0.5 for avoiding overfitting
  x = Dropout(0.5)(x)

  # last layer is fully connected layer with number of unit equals to number of class, 
  # use softmax activation because softmax output is probability distribution of vector inputs
  predictions = Dense(NUM_CLASSES, activation='softmax')(x)

  # model
  model = Model(inputs=base_model.input, outputs=predictions)

  # loop over all layers in the base model and freeze them so they
  # will *not* be updated during the first training process
  for layer in base_model.layers:
    layer.trainable = False

  # use Adam optimizer with learning rate is 0.0001
  opt = adam.Adam(lr=0.0001)

  # configure model with loss function is categorial cross entropy because of multi-class classification task
  # and metric is accuracy only before start training 
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  
  return model
