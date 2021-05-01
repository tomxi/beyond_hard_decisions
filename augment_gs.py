import json
import mirdata
import muda
import os
from tqdm import tqdm

giantsteps_key = mirdata.initialize('giantsteps_key', data_home='/scratch/qx244/data/gskey')
gs_data = giantsteps_key.load_tracks()

with open('/scratch/qx244/data/gskey/good_files.json', 'r') as fp:
    good_files = json.load(fp)

pitch_shifter = muda.deformers.LinearPitchShift(n_samples=12, lower=-5, upper=6)

for idx in tqdm(good_files.keys()):
    track = gs_data[idx]
    track_jams_path = os.path.join('/scratch/qx244/data/gskey/jams/', track.title + '.jams')

    #check if already augmented:
    if os.path.isfile('/scratch/qx244/data/gskey/augmentation/{}.11.jams'.format(track.title)):
        continue
        
    j_orig = muda.load_jam_audio(track_jams_path, track.audio_path)

    for i, jam_out in enumerate(pitch_shifter.transform(j_orig)):
        muda.save('/scratch/qx244/data/gskey/augmentation/{}.{:02d}.ogg'.format(track.title, i),
                  '/scratch/qx244/data/gskey/augmentation/{}.{:02d}.jams'.format(track.title, i),
                  jam_out)