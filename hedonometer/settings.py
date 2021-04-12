import torch

# SETTING: A string specifying the device to be used usually cuda:0 or cpu
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")