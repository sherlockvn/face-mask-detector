
import os
import pandas as pd 
from sklearn.model_selection import train_test_split
# configs
from src.config.config import IMG_DIR,PROJECT_PATH

# create dataset
def create_dataset():
  # remove old resources: dataset.csv, model.h5, service_weights.h5
  os.system("true > dataset.csv")
  os.system("true > " + PROJECT_PATH + "/models/model.h5")
  os.system("true > " + PROJECT_PATH + "/models/service_weights.h5")

  # init dataset
  dataset = []

  # read all files from image directories, then store it in dataset
  for folder in os.listdir(IMG_DIR):
      for filename in os.listdir(f'{IMG_DIR}/{folder}'):
          dataset.append((f'{folder}/{filename}', folder))

  # convert dataset to tabular form with column are filename, category
  df = pd.DataFrame(dataset, columns=['filename', 'category'])

  # divide the dataset into two sets: a training set and a test set, with an 80:20 ratio.
  df_train, df_test = train_test_split(df, random_state=42, stratify=df.category, test_size=.2)
  df_train['set'] = 'train'
  df_test['set'] = 'test'
  df = df_train.append(df_test)

  # convert data to csv file
  df.to_csv('dataset.csv', index=False)
  df.head()

  # read data
  df = pd.read_csv('dataset.csv')
  df.head()

  # create a new dataframe with a reset index
  train_df = df[df.set == 'train'].reset_index(drop=True)
  validate_df = df[df.set == 'test'].reset_index(drop=True)

  return [train_df, validate_df]
