from config import *
import os
from sklearn.model_selection import train_test_split
import pandas as pd 

# create dataset
def create_dataset():
  # remove all old contents first
  os.system("true > dataset.csv")
  os.system("true > model.h5")
  os.system("true > service_weights.h5")
  dataset = []
  # get all files in each image directories, then push it to dataset
  for fold in os.listdir(IMG_DIR):
      for filename in os.listdir(f'{IMG_DIR}/{fold}'):
          dataset.append((f'{fold}/{filename}', fold))
  # convert dataset to tabular form with column are filename, category
  df = pd.DataFrame(dataset, columns=['filename', 'category'])
  # split dataset into 2 set: training set and testint set with ratio 80:20
  df_train, df_test = train_test_split(df, random_state=42, stratify=df.category, test_size=.2)
  df_train['set'] = 'train'
  df_test['set'] = 'test'
  df = df_train.append(df_test)
  # convert data to csv file
  df.to_csv('dataset.csv', index=False)
  df.head()

  #read and prepare data
  df = pd.read_csv('dataset.csv')
  df.head()
  # generate new dataframe with index reset
  train_df = df[df.set == 'train'].reset_index(drop=True)
  validate_df = df[df.set == 'test'].reset_index(drop=True)
  return [train_df, validate_df]
