from config import MASKON_FOLDER, MASKOFF_FOLDER, MASKCHIN_FOLDER, MASKMOUTH_FOLDER
import cv2 as cv
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


def load_dataframe():
    '''
    comment here
    '''
    data_info = {
        "filename": [],
        "label": [],
        "target": [],
        "image": [],
    }

    with_mask = os.listdir(MASKON_FOLDER)
    without_mask = os.listdir(MASKOFF_FOLDER)
    mask_chin = os.listdir(MASKCHIN_FOLDER)
    mask_mouth = os.listdir(MASKMOUTH_FOLDER)

    print("[INFO] 1/4 loading dataset: mask chin dataset...")
    for filename in mask_chin:
        data_info["filename"].append(f"{MASKCHIN_FOLDER}/{filename}")
        data_info["label"].append(f"Mask only in the chin")
        data_info["target"].append(0)
        img = cv.cvtColor(cv.imread(f"{MASKCHIN_FOLDER}/{filename}"), cv.COLOR_BGR2RGB).flatten()
        data_info["image"].append(img)

    print("[INFO] 2/4 loading dataset: mask mouth dataset...")
    for filename in mask_mouth:
        data_info["filename"].append(f"{MASKMOUTH_FOLDER}/{filename}")
        data_info["label"].append(f"Mask below the nose")
        data_info["target"].append(1)
        img = cv.cvtColor(cv.imread(f"{MASKMOUTH_FOLDER}/{filename}"), cv.COLOR_BGR2RGB).flatten()
        data_info["image"].append(img)
    
    print("[INFO] 3/4 loading dataset: maskoff dataset...")
    for filename in without_mask:
        data_info["filename"].append(f"{MASKOFF_FOLDER}/{filename}")
        data_info["label"].append(f"Without Mask")
        data_info["target"].append(2)
        img = cv.cvtColor(cv.imread(f"{MASKOFF_FOLDER}/{filename}"), cv.COLOR_BGR2RGB).flatten()
        data_info["image"].append(img)
    
    
    print("[INFO] 4/4 loading dataset: maskon dataset...")
    for filename in with_mask:
        data_info["filename"].append(f"{MASKON_FOLDER}/{filename}")
        data_info["label"].append(f"With Mask")
        data_info["target"].append(3)
        img = cv.cvtColor(cv.imread(f"{MASKON_FOLDER}/{filename}"), cv.COLOR_BGR2RGB).flatten()
        data_info["image"].append(img)
    print("[INFO] DONE loaded all dataset!")
        
    dataframe = pd.DataFrame(data_info)

    return dataframe


def train_test(dataframe):
    '''
    '''
    X = list(dataframe["image"])
    y = list(dataframe["target"])

    return train_test_split(X, y, train_size=0.40, random_state=13)


def pca_model(x_train):
    '''
    '''
    pca = PCA(n_components=50)
    pca.fit(x_train)
    
    return pca

