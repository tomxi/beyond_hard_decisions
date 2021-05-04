import madmom
import numpy as np
import glob
import os
import sys

from tqdm import tqdm

aug_path = '/scratch/qx244/data/gskey/augmentation'
key_machine = madmom.features.key.CNNKeyRecognitionProcessor()

worker = int(sys.argv[1])
file_list = glob.glob(os.path.join(aug_path, '*.{:02d}.ogg'.format(worker)))

print(len(file_list))

for f in tqdm(file_list):
    output_path = os.path.splitext(f)[0] + '.npy'
    if os.path.exists(output_path):
        pass
    else:
        output = key_machine(f)
        # print(output)
        with open(output_path, 'wb') as o:
            np.save(o, output)