import os
import numpy as np
from tqdm import tqdm

NUCLEOTIDE_MAP = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1]
}

def one_hot_encode(sequence):
    return np.array([NUCLEOTIDE_MAP.get(nuc, [0, 0, 0, 0]) for nuc in sequence])

def load_data_from_folder(base_path, include_neg=False, include_synthetic=False):
    X, y = [], []

    # Load POS/ACC and POS/DON
    for class_name, label in [('ACC', 0), ('DON', 1)]:
        base_dir = os.path.join(base_path, 'POS', class_name)
        if not os.path.exists(base_dir):
            print(f"[WARN] POS folder not found: {base_dir}")
            continue

        for subtype in os.listdir(base_dir):
            subtype_path = os.path.join(base_dir, subtype)
            if not os.path.isdir(subtype_path):
                continue

            # Skip ADASYN folder always
            if subtype.upper().startswith("ADASYN"):
                continue

            # Only allow CAN and NC by default
            if not include_synthetic and subtype.upper() not in {"CAN", "NC"}:
                continue

            for root, _, files in os.walk(subtype_path):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, 'r') as f:
                            seq = f.read().strip()
                            if len(seq) == 600:
                                X.append(one_hot_encode(seq))
                                y.append(label)
                    except Exception as e:
                        print(f"[WARN] Skipping {fpath}: {e}")

    # Load NEG optionally
    if include_neg:
        for class_name, label in [('ACC', 2), ('DON', 3)]:
            neg_path = os.path.join(base_path, 'NEG', class_name)
            if not os.path.exists(neg_path):
                print(f"[WARN] NEG folder not found: {neg_path}")
                continue

            for fname in os.listdir(neg_path):
                fpath = os.path.join(neg_path, fname)
                try:
                    with open(fpath, 'r') as f:
                        seq = f.read().strip()
                        if len(seq) == 600:
                            X.append(one_hot_encode(seq))
                            y.append(label)
                except Exception as e:
                    print(f"[WARN] Skipping {fpath}: {e}")

    print(f"[INFO] Loaded {len(X)} sequences from {base_path} (include_neg={include_neg}, synthetic={include_synthetic})")
    return np.array(X), np.array(y)
