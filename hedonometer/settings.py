import torch

# SETTING: A string specifying the device to be used usually cuda:0 or cpu
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# SETTING: path to save the model in(includes extension)
SAVE_PATH = 'hedonometer/data/best_{}.bin'

# Data set used for training and validation it must contain the fields mainDescription(str),code(str)
DATA_SET_PATH = 'hedonometer/data/sentiments_m_cleaned.csv'

RANDOM_SEED = 123

# SETTING: path of the model to be used in the classifier
PRE_TRAINED_MODEL = "bert-base-portuguese-cased/"

# SETTING: Max length for the ids sequences given by the tokenizer
MAX_LEN = 512

# SETTING: Batch size
BATCH_SIZE = 4
