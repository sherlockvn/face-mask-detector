from config import *

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def fit_model(train_df, validate_df, model, callbacks_list, train_generator, validation_generator):
  #FIt model
  total_train = train_df.shape[0]
  total_validate = validate_df.shape[0]

  model.load_weights(CHECKPOINT_FILE)

  history = model.fit_generator(
      train_generator, 
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps=total_validate//BATCH_SIZE,
      steps_per_epoch=total_train//BATCH_SIZE,
      callbacks=callbacks_list
  )

  nb_samples = validate_df.shape[0]
  predict = model.predict_generator(validation_generator, steps=np.ceil(nb_samples/BATCH_SIZE))
  validate_df['pred'] = np.argmax(predict, axis=-1)
  label_map = dict((v,k) for k,v in train_generator.class_indices.items())
  validate_df['pred'] = validate_df['pred'].replace(label_map)

  print(classification_report(validate_df.category, validate_df.pred))

  print(confusion_matrix(validate_df.category, validate_df.pred))

  model.save('model_colab.h5')

