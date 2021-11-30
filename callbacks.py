from config import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# callbacks
callbacks_list = [
    ModelCheckpoint(filepath=CHECKPOINT_FILE, 
                    monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'),
    EarlyStopping(monitor='val_accuracy', patience=5),
    ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
]