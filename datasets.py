import json
import os
from glob import glob
import numpy as np
import jams
import librosa
from calibration import key_marg_add_logit, key_marg_simple_add, calibrate_key, new_cali_key, calibrate_root

class ChordsTestSet(object):   
    def __init__(self, data_home='/scratch/qx244/data/eric_chords_21sp'):
        self.data_home = data_home
        
        with open(os.path.join(data_home, 'index_test.json'), 'r') as j:
            self.index_dict = json.load(j)['id']
        
    def __len__(self):
        return len(self.index_dict)
    
    def audio_fp(self, idx):
        """idx: number between 0 and len of items-1"""
        return os.path.join(self.data_home, 'audio/{}.mp3'.format(self.index_dict[str(idx)]))
    
    def ref_fp(self, idx):
        """idx: number between 0 and len of items-1"""
        return os.path.join(self.data_home, 'references/{}.jams'.format(self.index_dict[str(idx)]))
    
    def npz_fp(self, idx):
        return os.path.join(self.data_home, 'crema_out/{}.npz'.format(self.index_dict[str(idx)]))

class RockCorpus(object):
    def __init__(self, data_home='/scratch/qx244/data/rock_corpus_v2-1/'):
        self.data_home = data_home
        self.titles = self._load_tracks()

    def __len__(self):
        return len(self.titles)

    def _load_tracks(self):
        anno_list = glob(os.path.join(self.data_home, 'timed/*.tcl'))
        titles = []
        for anno_p in anno_list:
            title = os.path.basename(anno_p).rsplit('_', 1)[0]
            if title not in titles:
                titles.append(title)
        titles.sort()
        return titles

    def audio_path(self, idx):
        title = self.titles[idx]
        out = os.path.join(self.data_home, 'rs_audio_masters_mp3/{}.mp3'.format(title))
        assert os.path.exists(out)
        return out

    def tdc_anno_path(self, idx):
        title = self.titles[idx]
        out = os.path.join(self.data_home, 'timed/{}_tdc.tcl'.format(title))
        assert os.path.exists(out)
        return out

    def dt_anno_path(self, idx):
        title = self.titles[idx]
        out = os.path.join(self.data_home, 'timed/{}_dt.tcl'.format(title))
        assert os.path.exists(out)
        return out

    def proc_path(self, idx):
        title = self.titles[idx]
        out = os.path.join(self.data_home, 'proc/{}.npz'.format(title))
        return out

    def jams_path(self, idx):
        title = self.titles[idx]
        out = os.path.join(self.data_home, 'jams/{}.jams'.format(title))
        return out

    def track_data(self, idx):
        out = {}

        data = np.load(self.proc_path(idx), allow_pickle=True)
        key_output = data['key']
        root_output = data['chord'].item()['chord_root']

        out['raw_key'] = np.roll(key_marg_add_logit(key_output), -3, axis=1)
        out['raw_key_new'] = np.roll(key_marg_simple_add(key_output), -3, axis=1)
        out['raw_root'] = root_output

        out['calib_key'] = calibrate_key(key_output)
        out['calib_key_new'] = new_cali_key(key_output)
        out['calib_root'] = calibrate_root(root_output)

        out['hard_key'] = np.eye(12)[np.argmax(out['raw_key'], axis=1)]
        out['hard_key_new'] = np.eye(12)[np.argmax(out['raw_key_new'], axis=1)]
        out['hard_root'] = np.eye(13)[np.argmax(out['raw_root'], axis=1)]
        return out

    def ann_data(self, idx):
        jam = jams.load(self.jams_path(idx))
        frame_period = 4096 / 44100

        anns = jam.search(namespace='pitch_class')
        key_mats = []
        root_mats = []
        for a in anns:
            num_frames = (a.duration - frame_period/2) / frame_period
            f_times = np.arange(np.floor(num_frames)) * frame_period + frame_period/2

            frame_list = a.to_samples(f_times)
            frame_rel_root = []
            frame_key = []
            for frame in frame_list:
                if len(frame) == 0:
                    frame_rel_root.append(12)
                    frame_key.append(12)
                else:
                    frame_key.append(librosa.note_to_midi(frame[0]['tonic']) % 12)
                    frame_rel_root.append(frame[0]['pitch'])

            key_mats.append(np.eye(13)[frame_key])
            root_mats.append(np.eye(13)[frame_rel_root])
        
        return {'key': np.mean(key_mats, axis=0), 
                'root': np.mean(root_mats, axis=0)}


class Billboard(object):
    def __init__(self, data_home='/scratch/qx244/data/billboard/'):
        self.data_home = data_home