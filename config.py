from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# session = InteractiveSession(config=config)

# parameters
CHECKPOINT_FILE = './service_weights.h5'
MODEL_SAVE = 'service_type_model.h5'
FAST_RUN = False
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
IMG_DIR = 'dataset/imagens/'
BATCH_SIZE = 32
NUM_CLASSES = 4

EPOCHS=20
DATASET = "dataset/imagens/"
MASKON_FOLDER = DATASET + "maskon"
MASKOFF_FOLDER = DATASET + "maskoff"
MASKCHIN_FOLDER = DATASET + "maskchin"
MASKMOUTH_FOLDER = DATASET + "maskmouth"

CAFFE_PROTO_FILE = "caffe_face_detector/deploy.prototxt.txt"
MODEL_FILE = "caffe_face_detector/res10_300x300_ssd_iter_140000.caffemodel"
CONFIDENCE = 0.90