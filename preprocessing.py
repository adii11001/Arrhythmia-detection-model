import torch
import wfdb
import numpy as np

class Config:
    WINDOW_SIZE = 256
    SAMPLE_RATE = 360
    AAMI_CLASSES = ['N', 'S', 'V', 'F', 'Q']

    AAMI_MAP = {
        # Normal (N)
        'N': 'N',  # Normal beat
        'L': 'N',  # Left bundle branch block beat
        'R': 'N',  # Right bundle branch block beat
        'e': 'N',  # Atrial escape beat
        'j': 'N',  # Nodal (junctional) escape beat

        # Supraventricular Ectopic (S)
        'A': 'S',  # Atrial premature beat
        'a': 'S',  # Aberrated atrial premature beat
        'J': 'S',  # Nodal (junctional) premature beat
        'S': 'S',  # Supraventricular premature beat

        # Ventricular Ectopic (V)
        'V': 'V',  # Premature ventricular contraction
        'E': 'V',  # Ventricular escape beat

        # Fusion (F)
        'F': 'F',  # Fusion of ventricular and normal beat

        # Unknown/Unclassifiable (Q)
        '/': 'Q',  # Paced beat
        'f': 'Q',  # Fusion of paced and normal beat
        'Q': 'Q',  # Unclassifiable beat
    }

def loading_and_segmenting(window_size=256):
    filepath = "./mit-bih-arrhythmia-database-1.0.0/"
    segments = []
    labels = []
    half_window_size = window_size // 2
    LABEL_TO_IDX = {i: Config.AAMI_CLASSES.index(Config.AAMI_MAP[i]) for i in Config.AAMI_MAP}

    with open(filepath + "RECORDS") as f:
        records = f.read().splitlines()

    for r in records:
        record = wfdb.rdrecord(filepath + r)
        annotation = wfdb.rdann(filepath + r, "atr")
        signal = record.p_signal[:, 0]

        for i in range(len(annotation.sample)):
            r_peak = annotation.sample[i]
            if annotation.symbol[i] in Config.AAMI_MAP:
                if r_peak >= half_window_size and len(signal) - r_peak >= half_window_size:
                    segment = signal[r_peak - half_window_size: r_peak + half_window_size]
                    segment = (segment - segment.mean()) / (segment.std() + 1e-8)
                    segments.append(segment)
                    labels.append(LABEL_TO_IDX[annotation.symbol[i]])

    return np.array(segments, dtype=np.float32), np.array(labels, dtype=np.int64)
