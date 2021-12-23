import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from plot_keras_history import show_history, plot_history
# configs
from src.config.config import EPOCHS, BATCH_SIZE, PROJECT_PATH

# fit model
def fit_model(train_df, validate_df, model, callbacks_list, train_generator, validation_generator):
  total_train = train_df.shape[0]
  total_validate = validate_df.shape[0]

  # model.load_weights(CHECKPOINT_FILE)
  print("[INFO] training head...")
  history = model.fit(
      train_generator, 
      epochs=EPOCHS,
      validation_data=validation_generator,
      validation_steps=total_validate//BATCH_SIZE,
      steps_per_epoch=total_train//BATCH_SIZE,
      callbacks=callbacks_list
  )

  nb_samples = validate_df.shape[0]

  # using model to predicting
  predict = model.predict(validation_generator, steps=np.ceil(nb_samples/BATCH_SIZE))
  validate_df['pred'] = np.argmax(predict, axis=-1)

  # label predictions
  label_map = dict((v,k) for k,v in train_generator.class_indices.items())
  validate_df['pred'] = validate_df['pred'].replace(label_map)

  # print result
  print(classification_report(validate_df.category, validate_df.pred))
  print(confusion_matrix(validate_df.category, validate_df.pred))

  # save model
  model.save(PROJECT_PATH + "/models/model.h5")

  # show and save history report
  show_history(history)
  plot_history(history, path= PROJECT_PATH + "/reports/history_report.png")

