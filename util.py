import numpy as np
import os
import json
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

class KeysTestSet(object):

    def __init__(self, data_home='/scratch/qx244/data/gskey'):
        self.data_home = data_home

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
