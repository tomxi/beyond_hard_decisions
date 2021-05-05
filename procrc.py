import madmom
import numpy as np
import glob
import os
import sys
import util as ut
import crema

from tqdm import tqdm

rc = ut.RockCorpus()
key_machine = madmom.features.key.CNNKeyRecognitionProcessor()
chord_model = crema.models.chord.ChordModel()

worker = int(sys.argv[1]) # 0-4
idx_list = [idx for idx in range(len(rc)) if idx % 5 == worker]

for i in tqdm(idx_list):
    output_path = rc.proc_path(i)
    if os.path.exists(output_path):
        pass
    else:
        key_output = key_machine(rc.audio_path(i))
        crema_output = chord_model.outputs(filename=rc.audio_path(i))
        with open(output_path, 'wb') as o:
            np.savez(o, key=key_output, chord=crema_output)