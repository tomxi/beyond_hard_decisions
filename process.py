import crema
import numpy as np
import jams

def chord_file_processor(audio_fp, ref_fp):
    """
    Returns the processed numpy arrays processed by crema
    """
    chord_model = crema.models.chord.ChordModel()
    crema_output = chord_model.outputs(filename=audio_fp)
    c_pump = chord_model.pump['chord_struct']
    
    ref_jams = jams.load(ref_fp)
    target_dist = c_pump.transform(ref_jams)
    
    hard_label = np.squeeze(target_dist['chord_struct/root'])
    soft_out = crema_output['chord_root']
    
    return hard_label, soft_out
    