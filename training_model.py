from keras import callbacks
from create_dataset import create_dataset
from base_model import create_base_model
import callbacks
from generator import generator
from fit_model import fit_model
from dataframe import *
# preprocess dataset
train_df, validate_df = create_dataset()
print("[INFO] create dataset successfully!")
# init base model
model = create_base_model();
print("[INFO] create base model successfully!")
# callbacks list
callbacks_list = callbacks.callbacks_list
# generator
train_generator, validation_generator = generator(train_df, validate_df);
print("[INFO] generator ok!")
# init fit_model
fit_model(train_df, validate_df, model, callbacks_list, train_generator, validation_generator)
# load frame
dataframe = load_dataframe() 

X_train, X_test, y_train, y_test = train_test(dataframe) 
pca = pca_model(X_train) 








