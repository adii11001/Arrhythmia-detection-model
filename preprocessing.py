import torch
from torch.utils.data import WeightedRandomSampler, Dataset, DataLoader, random_split
import wfdb
import numpy as np

class Config:
    """
    Class for storing configuration variables
    """
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

class ECGDataset(Dataset):
    """
    Class for ECG Dataset
    """
    def __init__(self, segments, labels):
        self.segments = torch.tensor(segments, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.segments[idx], self.labels[idx]

def loading_and_segmenting(window_size=256):
    """
    Function to load and segment the ecg signals from the MIT-BIH-Arrhythmia Dataset
    :param window_size:int
    :return: np.array
    """
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
                if r_peak >= half_window_size and r_peak + half_window_size <= len(signal):
                    segment = signal[r_peak - half_window_size: r_peak + half_window_size]
                    segment = (segment - segment.mean()) / (segment.std() + 1e-8)
                    segments.append(segment)
                    labels.append(LABEL_TO_IDX[annotation.symbol[i]])

    return np.array(segments, dtype=np.float32), np.array(labels, dtype=np.int64)

def balanced_loader(dataset: ECGDataset, batch_size=32):
    """
    Applies WeightRandomSampler on the dataset and creates a DataLoader to ensure that the DataLader is balanced.
    :param dataset: ECGDataset
    :param batch_size: int
    :return: torch.utils.data.Dataloader
    """

    num_classes = len(Config.AAMI_CLASSES)
    class_sample_count = np.zeros(num_classes)
    for t in dataset.labels.numpy():
        class_sample_count[t] += 1
    class_sample_count = np.where(class_sample_count == 0, 1, class_sample_count)
    weight = 1. / class_sample_count
    samples_weight = torch.tensor([weight[t] for t in dataset.labels.numpy()])

    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
