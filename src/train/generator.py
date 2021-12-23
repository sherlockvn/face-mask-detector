import random
import numpy as np
# configs
from src.config.config import IMG_DIR, IMAGE_SIZE, BATCH_SIZE
# keras api
from keras.preprocessing.image import ImageDataGenerator

# const
VARIABILITY = 8
COLOR_MODE = 'rgb'

# add random noise to an image
def add_noise(img):
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

# generator
def generator(train_df, validate_df):
  # generate batches of image data with configuration below, also add noise
  train_datagen = ImageDataGenerator(
      brightness_range=[0.2, 1.6],
      rescale=1. / 255, # for converting rgb coefficients in 0-255 in original images to values between 0 and 1
      rotation_range=0, 
      width_shift_range=0.1,
      height_shift_range=0.1, 
      shear_range=0.2, 
      zoom_range=0.2,
      horizontal_flip=True, 
      fill_mode="nearest",
      preprocessing_function=add_noise,
  )

  train_generator = train_datagen.flow_from_dataframe(
      train_df, 
      IMG_DIR, 
      x_col='filename',
      y_col='category',
      target_size=IMAGE_SIZE,
      color_mode=COLOR_MODE,
      class_mode='categorical',
      batch_size=BATCH_SIZE
  )

  validation_datagen = ImageDataGenerator(rescale=1./255)
  validation_generator = validation_datagen.flow_from_dataframe(
      validate_df, 
      IMG_DIR,
      x_col='filename',
      y_col='category',
      target_size=IMAGE_SIZE,
      color_mode=COLOR_MODE,
      class_mode='categorical',
      shuffle=False,
      batch_size=BATCH_SIZE
  )
  return [train_generator, validation_generator]