# The path to where the images and labels are located
DATA_PATH = '.'

# The path to the text file that contains the names of the files to be used for training
TRAIN_FILE = './data/birds_train.txt'

# The path to the text file that contains the names of the files to be used for testing
TEST_FILE = './data/birds_test.txt'

# The path to the text file that contains the names of the classes
NAME_FILE = './data/names.txt'

# The name of the directory of where to save the weights
SAVE_DIR = 'v1'

# The dimensions to resize the imanges to before passing it through the network
IMG_INPUT_SIZE = [512, 512]

# The number of cells along each axis
GRID_SIZE = 16

# The total number of classes in the dataset
NUM_CLASSES = 404

# The batch size to use for training
BATCH_SIZE = 8

# Number of images to reserve for validation
VALID_SIZE = 4096

# The number of epochs to train for
EPOCHS = 10

# The learning rate to use while training
LEARNING_RATE = 1

# The scale factor for the negative samples during training
NEG_SAMPLE_SCALE = 1

# The minimum confidence required for a prediction to be considered
NMS_THRESHOLD = 0.65

# The maximum overlap allowed  between to predictions
IOU_THRESHOLD = 0.7

# Small value to escape 0 errors
EPSILON = 1e-5

# Seed used for the random library
SEED = 13290

# Dropout probability
DROPOUT = 0.2