import numpy as np
import os
import json
import jams
from glob import glob
from scipy.optimize import minimize_scalar

def better_track_nll(beta, hard_label, soft_out):
    """ just summing frame_nll across the whole track
    Param:
        beta: inverse temp in [0,1]
        hard_label: vector of labels, index style
        soft_out: matrix with shape: len(hard_label) x # of classes
    Returns:
        Aggregated frame_nll
    """
    z_mat = np.log(soft_out) # time x classes
    logsumexp_vec = np.log(np.sum(np.exp(beta*z_mat), axis=-1, keepdims=True)) # find a more stable version, shape = t
    per_frame_nll = -beta*np.take_along_axis(z_mat, hard_label[:,np.newaxis], axis=1) + logsumexp_vec
    return np.mean(per_frame_nll)

def better_calibrate(beta, raw_conf):
    """ Calibrate the raw output with parameter beta
    params:
        beta: inverse temprature
        raw_conf: np.array of uncalibrated confidence, shape: # of entries x # of classes
    returns:
        calibrated_conf: same shape as raw_out, but caßlibrated
    """
    unnormalized = raw_conf ** beta
    return unnormalized / np.sum(unnormalized, axis=1, keepdims=True)

def softmax(mat):
    """shape: x * 12"""
    e = np.exp(mat)
    return e / np.sum(e, axis=1, keepdims=True)

def key_marg_simple_add(key_output):
    """bin 0 is A, bin 1 is Bb, bin 11 is G#"""
    return key_output[:, :12] + key_output[:, 12:]

def key_marg_add_logit(key_output):
    """bin 0 is A, bin 1 is Bb, bin 11 is G#"""
    z = np.log(key_output)
    return softmax(z[:, :12] + z[:, 12:])

def key_marg_hard_max(key_output):
    """bin 0 is A"""
    zmax = np.maximum(key_output[:, :12], key_output[:, 12:])
    zmax /= zmax.sum(axis=1, keepdims=True)
    return zmax

def calibrate_root(raw_conf):
    return better_calibrate(0.7956625610036311, raw_conf)

def calibrate_key(key_output):
    """Calibrate and roll towards 0 = C"""
    calibrated = better_calibrate(0.7806331984345791, key_marg_add_logit(key_output))
    return np.roll(calibrated, -3, axis=1)

def relative_root(key_output, root_output):
    output = np.zeros(root_output.shape)
    
    root_only = root_output[:, :12] # no N class
    for tonic in range(12):
        rel_root = np.roll(root_only, -tonic, axis=1)
        output[:, :12] += rel_root * key_output[:, tonic]
    
    output[:, 12] = root_output[:, 12]
    return output

class CalibrationBenchmark(object):
    
    def __init__(self, data, label, m=15):
        self.data = data
        self.label = label
        self.m = m
        
        self.binned_index = []
        self.conf = []
        self.pred = []
        self.calib_res = None
        
        self.create_bins()
        
    def create_bins(self):
        """ Bin the samples according to their confidence
        Params:
            data: np.array with shape: (#entries, #classes)
            m: int > 1; number of bins; default is 15 bins
        Returns:
            binned_index: a list of lists: len(binned_index) = m, and each of these m lists contain the indicies of 
            the original dataset that has their output confidence in the range of the associated bin. 
        """
        # create empty list of lists
        binned_index = [[] for _ in range(self.m)]
        entry_count, _ = self.data.shape

        # get the confidence of the predicted class
        self.conf = self.data.max(axis=1)
        self.pred = self.data.argmax(axis=1)
        # strategy: expand(multiply) the conf vector by a factor of m, then use flooring
        tentative_bins = np.floor(self.conf * self.m).astype(int)
        for i in range(entry_count):
            if tentative_bins[i] != self.m:
                binned_index[tentative_bins[i]].append(i)
            else: # when confidence is exactly 1, put in the last bin
                binned_index[self.m-1].append(i)
        self.binned_index = binned_index
        
    def hist(self):
        return np.array([len(self.binned_index[i]) for i in range(len(self.binned_index))])
    
    def bin_accuracy(self):
        result = np.empty((self.m,))
        
        for i in range(self.m):
            if len(self.binned_index[i]) == 0:
                result[i] = 0
            else:
                result[i] = np.mean(self.pred[self.binned_index[i]] == self.label[self.binned_index[i]])
        return result
    
    def bin_confidence(self):
        result = np.empty((self.m,))
            
        for i in range(self.m):
            if len(self.binned_index[i]) == 0:
                result[i] = 0
            else:
                result[i] = np.mean(self.conf[self.binned_index[i]])
        return result
    
    def ece(self):
        error = self.bin_accuracy() - self.bin_confidence()
        weight = self.hist() / self.data.shape[0]
        return np.dot(np.abs(error), weight)

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
        out['raw_root'] = root_output

        out['calib_key'] = calibrate_key(key_output)
        out['calib_root'] = calibrate_root(root_output)

        out['hard_key'] = np.eye(12)[np.argmax(out['raw_key'], axis=1)]
        out['hard_root'] = np.eye(13)[np.argmax(out['raw_root'], axis=1)]
        return out

    def ann_data(self, idx):
        jam = jams.load(self.jams_path(idx))
        frame_period = 4096 / 44100

        anns = jam.search(namespace='pitch_class')
        for a in anns:
            num_frames = (a.duration - frame_period/2) / frame_period
            f_times = np.arange(np.floor(num_frames)) * frame_period + frame_period/2

            return a.to_samples(f_times)