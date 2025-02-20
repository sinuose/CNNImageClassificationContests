# Contest Submission for:
# https://www.kaggle.com/c/diabetic-retinopathy-detection/data

from hdr.EfficientNet import *
from hdr.preprocess import *
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

# Are you training??
TTTTTT = 0

# version: width, depth, res, dropout rate
efficient_net_config = {
"b0" : (1.0, 1.0, 224, 0.2),
"b1" : (1.0, 1.1, 240, 0.2),
"b2" : (1.1, 1.2, 260, 0.3),
"b3" : (1.2, 1.4, 300, 0.3),
"b4" : (1.4, 1.8, 380, 0.4),
"b5" : (1.6, 2.2, 456, 0.4),
"b6" : (1.8, 2.6, 528, 0.5),
"b7" : (2.0, 3.1, 600, 0.5)
}

if __name__ == "__main__":

    print("Program Start")
    # version readout
    version = 'b0'
    width_mult, depth_mult, res, dropout_rate = efficient_net_config[version]
    print(f"Running EfficientNet with params for Version: {version}")
    # generate version
    net = EfficientNet(width_mult, depth_mult, dropout_rate)

    #if input("\nNetwork successfully loaded. Proceed with program? (Y/N)\n\n>:") != "Y":
    #    quit()
    
    DATA_DIR = os.getcwd() + "//DiabeticRetinopathy//resources"  
    # This is the dataset that we have
    # dataset of all images, and patient IDs
    dataset = DiabeticRetinopathyDataset(DATA_DIR + "//data//sample")

