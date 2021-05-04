import madmom
import numpy as np
import glob
import os
import sys

from tqdm import tqdm

aug_path = '/scratch/qx244/data/gskey/augmentation'
# key_machine = madmom.features.key.CNNKeyRecognitionProcessor()

worker = sys.argv[1]
file_list = glob.glob(os.path.join(aug_path, '*.{:02d}.ogg'.format(worker)))

print(len(file_list))