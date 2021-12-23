# configs
from src.config.config import CHECKPOINT_FILE
# keras api
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# callbacks list include 
# save model to checkpoint file, 
# stop training when accuracy has stopped improving
# reduce learning rate when accuracy has stopped improving with factor of 0.5, minimum is 0.00001
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