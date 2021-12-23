import callbacks
from create_dataset import create_dataset
from model import create_base_model
from fit_model import fit_model
from generator import generator
from dataframe import *
# keras api
from tensorflow.keras import callbacks

# preprocess dataset
train_df, test_df = create_dataset()
print("[INFO] create dataset successfully!")

# init base model
model = create_base_model()
print("[INFO] create base model successfully!")

# init callbacks list
callbacks_list = callbacks.callbacks_list
print("[INFO] callbacks list initialized successfully!")

# generator
train_generator, train_generator = generator(train_df, test_df)
print("[INFO] generator ok!")

# init fit_model
# callbacks list will then be called at each stage of the training
fit_model(train_df, test_df, model, callbacks_list, train_generator, train_generator)
print("[INFO] fit model initialized successfully!")

# load frame
dataframe = load_dataframe() 
print("[INFO] dataframe is loaded!")

# train test
X_train, X_test, y_train, y_test = train_test(dataframe) 

# pca model
pca = pca_model(X_train) 