import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from plot_keras_history import show_history, plot_history
# configs
from src.config.config import EPOCHS, BATCH_SIZE

# fit model
def fit_model(train_df, test_df, model, callbacks_list, train_generator, test_generator):
  total_train = train_df.shape[0]
  total_test = test_df.shape[0]

  # model.load_weights(CHECKPOINT_FILE)
  print("[INFO] training head...")
  history = model.fit(
      train_generator, 
      epochs=EPOCHS,
      validation_data=test_generator,
      validation_steps=total_test//BATCH_SIZE,
      steps_per_epoch=total_train//BATCH_SIZE,
      callbacks=callbacks_list
  )

  nb_samples = test_df.shape[0]

  # using model to predicting
  predict = model.predict(test_generator, steps=np.ceil(nb_samples/BATCH_SIZE))
  test_df['pred'] = np.argmax(predict, axis=-1)

  # label predictions
  label_map = dict((v,k) for k,v in train_generator.class_indices.items())
  test_df['pred'] = test_df['pred'].replace(label_map)

  # print result
  print(classification_report(test_df.category, test_df.pred))
  print(confusion_matrix(test_df.category, test_df.pred))

  # save model
  model.save(PROJECT_PATH + "/models/model.h5")

  # show and save history report
  show_history(history)
  plot_history(history, path= PROJECT_PATH + "/reports/history_report.png")

