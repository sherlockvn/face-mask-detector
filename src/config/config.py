import os

# path
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

# parameters
CHECKPOINT_FILE = PROJECT_PATH + "/models/service_weights.h5"
MODEL_SAVE = PROJECT_PATH + "/models/service_type_model.h5"
FAST_RUN = False
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
IMG_DIR = PROJECT_PATH + "/dataset/imagens/"
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 50

# dataset
MASKON_FOLDER = IMG_DIR + "maskon"
MASKOFF_FOLDER = IMG_DIR + "maskoff"
MASKCHIN_FOLDER = IMG_DIR + "maskchin"
MASKMOUTH_FOLDER = IMG_DIR + "maskmouth"

# CAFFE model
CAFFE_PROTO_FILE = PROJECT_PATH + "/caffe_face_detector/deploy.prototxt.txt"
MODEL_FILE = PROJECT_PATH + "/caffe_face_detector/res10_300x300_ssd_iter_140000.caffemodel"
CONFIDENCE = 0.7
