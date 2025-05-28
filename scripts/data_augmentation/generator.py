import os
import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import NotFittedError
from imblearn.over_sampling import SMOTE

def load_sequences_from_folder(folder_path):
    sequences = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), 'r') as file:
                sequences.append(file.read().strip())
    print(f"Loaded {len(sequences)} sequences from {folder_path}")
    return sequences

def sequence_to_onehot(sequences, alphabet="ACGT"):
    encoder = OneHotEncoder(categories=[list(alphabet)], sparse=False)
    one_hot_sequences = []
    for seq in sequences:
        seq_array = np.array(list(seq)).reshape(-1, 1)
        one_hot_seq = encoder.fit_transform(seq_array)
        one_hot_sequences.append(one_hot_seq)
    return np.array(one_hot_sequences).reshape(len(sequences), -1), encoder

def onehot_to_sequence(onehot_data, encoder):
    sequences = []
    for onehot_seq in onehot_data:
        decoded = encoder.inverse_transform(onehot_seq.reshape(-1, len(encoder.categories_[0])))
        sequences.append(''.join(decoded.flatten()))
    return sequences

def apply_adasyn(X_non_canonical, target_count):
    if len(X_non_canonical) < 2:
        raise ValueError("ADASYN requires at least 2 non-canonical sequences.")

    dummy = np.zeros((1, X_non_canonical.shape[1]))
    X_combined = np.vstack([X_non_canonical, dummy])
    y_combined = np.array([0] * len(X_non_canonical) + [1])

    n_neighbors = min(5, len(X_non_canonical) - 1)
    adasyn = ADASYN(sampling_strategy={0: target_count}, random_state=42, n_neighbors=n_neighbors)

    try:
        X_resampled, y_resampled = adasyn.fit_resample(X_combined, y_combined)
    except RuntimeError as e:
        raise RuntimeError(f"[ADASYN ERROR] {str(e)}. You may need more samples or a lower target_count.") from e

    return X_resampled[y_resampled == 0]

def apply_smote(X_non_canonical, target_count):
    if len(X_non_canonical) < 2:
        raise ValueError("SMOTE requires at least 2 non-canonical sequences.")
    
    dummy = np.zeros((1, X_non_canonical.shape[1]))
    X_combined = np.vstack([X_non_canonical, dummy])
    y_combined = np.array([0] * len(X_non_canonical) + [1])

    n_neighbors = min(5, len(X_non_canonical) - 1)
    smote = SMOTE(sampling_strategy={0: target_count}, random_state=42, k_neighbors=n_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X_combined, y_combined)

    return X_resampled[y_resampled == 0]

def save_sequences_to_folder(folder_path, sequences, prefix="synthetic"):
    os.makedirs(folder_path, exist_ok=True)
    for i, seq in enumerate(sequences):
        with open(os.path.join(folder_path, f"{prefix}_{i+1}.txt"), 'w') as f:
            f.write(seq)
    print(f"Saved {len(sequences)} sequences to {folder_path}")
