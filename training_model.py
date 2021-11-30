from keras import callbacks
from create_dataset import create_dataset
from base_model import create_base_model
import callbacks
from generator import generator
from fit_model import fit_model
from dataframe import load_dataframe
# preprocess dataset
train_df, validate_df = create_dataset()
# init base model
model = create_base_model();
# callbacks list
callbacks_list = callbacks.callbacks_list
# generator
train_generator, validation_generator = generator(train_df, validate_df);
# init fit_model
fit_model(train_df, validate_df, model, callbacks_list, train_generator, validation_generator)
# load frame
load_dataframe()








