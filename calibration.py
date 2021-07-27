import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

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

def new_cali_key(output, b=1.1880162333380975):
    calibrated_full = better_calibrate(b, output)
    calibrated = calibrated_full[:, :12] + calibrated_full[:, 12:]
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

    def plot_rd(self, fname=None):
        def x_ticks(m=15):
            posts = np.linspace(0,1,num=m+1)
            bin_center = (posts[:-1] + posts[1:]) / 2
            width = posts[1]
            return bin_center, width
        
        hist = self.hist()
        acc = self.bin_accuracy()
        conf = self.bin_confidence()
        
        x, width = x_ticks(m=self.m)
        fig, axs = plt.subplots(1, 2 ,figsize=(8, 4))
        ax, axh = axs
        ax.plot([0.01,0.99], [0.01,0.99], ':')
        ax.bar(x, acc, width, edgecolor='k', label='output')
        ax.bar(x, conf - acc, width, acc, edgecolor='r', fill=False, hatch='\\', label='gap')

        ax.axis('equal')
        ax.set(xlabel='Confidence', ylabel='Accuracy', xlim=(0,1), ylim=(0,1),
               title='Reliability Diagram')
        ax.legend()

        axh.bar(x, hist / hist.sum(), width, edgecolor='k')
        axh.axis('equal')
        axh.set(xlabel='Confidence', ylabel='Prevalence', xlim=(0,1), ylim=(0,1),
               title='Confidence Histogram')

        if fname: 
            fig.savefig(fname)
        plt.show()
