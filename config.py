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
IMAGE_CHANNELS = 1
IMG_DIR = 'dataset/'
BATCH_SIZE = 32
NUM_CLASSES = 4

epochs=50 if FAST_RUN else 50
DATASET = "dataset/imagens/"
MASKON_FOLDER = DATASET + "maskon"
MASKOFF_FOLDER = DATASET + "maskoff"
MASKCHIN_FOLDER = DATASET + "maskchin"
MASKMOUTH_FOLDER = DATASET + "maskmouth"