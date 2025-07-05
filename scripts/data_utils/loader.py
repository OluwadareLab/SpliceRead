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

def load_sequences_from_folder(folder_path, label, show_progress=False, desc="Loading", sequence_length=600):
    data, labels = [], []
    files = os.listdir(folder_path)
    iterator = tqdm(files, desc=desc, disable=not show_progress)
    for file_name in iterator:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == sequence_length: 
                    data.append(one_hot_encode(line))
                    labels.append(label)
    return np.array(data), np.array(labels)

def load_data(base_path, adasyn_subdir, show_progress=False, sequence_length=600):
    pos_path = os.path.join(base_path, 'POS')
    acc_path = os.path.join(pos_path, 'ACC')
    don_path = os.path.join(pos_path, 'DON')

    print("[INFO] Loading ACC/CAN data...")
    acc_can_data, acc_can_labels = load_sequences_from_folder(os.path.join(acc_path, 'CAN'), 0, show_progress, "Loading ACC/CAN", sequence_length)
    print("[INFO] Loading ACC/NC data...")
    acc_nc_data, acc_nc_labels = load_sequences_from_folder(os.path.join(acc_path, 'NC'), 1, show_progress, "Loading ACC/NC", sequence_length)
    print(f"[INFO] Loading ACC/ADASYN/{adasyn_subdir} data...")
    acc_syn_data, acc_syn_labels = load_sequences_from_folder(os.path.join(acc_path, 'ADASYN', adasyn_subdir), 2, show_progress, f"Loading ACC/ADASYN/{adasyn_subdir}", sequence_length)

    print("[INFO] Loading DON/CAN data...")
    don_can_data, don_can_labels = load_sequences_from_folder(os.path.join(don_path, 'CAN'), 3, show_progress, "Loading DON/CAN", sequence_length)
    print("[INFO] Loading DON/NC data...")
    don_nc_data, don_nc_labels = load_sequences_from_folder(os.path.join(don_path, 'NC'), 4, show_progress, "Loading DON/NC", sequence_length)
    print(f"[INFO] Loading DON/ADASYN/{adasyn_subdir} data...")
    don_syn_data, don_syn_labels = load_sequences_from_folder(os.path.join(don_path, 'ADASYN', adasyn_subdir), 5, show_progress, f"Loading DON/ADASYN/{adasyn_subdir}", sequence_length)

    data = np.concatenate([acc_can_data, acc_nc_data, acc_syn_data, don_can_data, don_nc_data, don_syn_data])
    labels = np.concatenate([acc_can_labels, acc_nc_labels, acc_syn_labels, don_can_labels, don_nc_labels, don_syn_labels])

    print(f"[INFO] Loaded data shapes: {data.shape}, {labels.shape}")
    print(f"[INFO] Class distribution: {np.bincount(labels)}")
    
    return data, labels

def load_data_with_negatives(base_path, adasyn_subdir, show_progress=False, sequence_length=600):
    
    pos_path = os.path.join(base_path, 'POS')
    neg_path = os.path.join(base_path, 'NEG')
    acc_path = os.path.join(pos_path, 'ACC')
    don_path = os.path.join(pos_path, 'DON')

    print("[INFO] Loading ACC/CAN data...")
    acc_can_data, acc_can_labels = load_sequences_from_folder(os.path.join(acc_path, 'CAN'), 0, show_progress, "Loading ACC/CAN", sequence_length)
    
    print("[INFO] Loading ACC/NC data...")
    acc_nc_data, acc_nc_labels = load_sequences_from_folder(os.path.join(acc_path, 'NC'), 1, show_progress, "Loading ACC/NC", sequence_length)
    
    acc_syn_data, acc_syn_labels = np.array([]), np.array([])
    if adasyn_subdir:
        adasyn_acc_path = os.path.join(acc_path, 'ADASYN', adasyn_subdir)
        if os.path.exists(adasyn_acc_path):
            print(f"[INFO] Loading ACC/ADASYN/{adasyn_subdir} data (merging with ACC/NC)...")
            acc_syn_data, acc_syn_labels = load_sequences_from_folder(adasyn_acc_path, 1, show_progress, f"Loading ACC/ADASYN/{adasyn_subdir}", sequence_length)

    print("[INFO] Loading DON/CAN data...")
    don_can_data, don_can_labels = load_sequences_from_folder(os.path.join(don_path, 'CAN'), 2, show_progress, "Loading DON/CAN", sequence_length)
    
    print("[INFO] Loading DON/NC data...")
    don_nc_data, don_nc_labels = load_sequences_from_folder(os.path.join(don_path, 'NC'), 3, show_progress, "Loading DON/NC", sequence_length)
    
    don_syn_data, don_syn_labels = np.array([]), np.array([])
    if adasyn_subdir:
        adasyn_don_path = os.path.join(don_path, 'ADASYN', adasyn_subdir)
        if os.path.exists(adasyn_don_path):
            print(f"[INFO] Loading DON/ADASYN/{adasyn_subdir} data (merging with DON/NC)...")
            don_syn_data, don_syn_labels = load_sequences_from_folder(adasyn_don_path, 3, show_progress, f"Loading DON/ADASYN/{adasyn_subdir}", sequence_length)

    print("[INFO] Loading NEG sequences (non-splice sites)...")
    neg_acc_data, neg_acc_labels = load_sequences_from_folder(os.path.join(neg_path, 'ACC'), 4, show_progress, "Loading NEG/ACC", sequence_length)
    neg_don_data, neg_don_labels = load_sequences_from_folder(os.path.join(neg_path, 'DON'), 4, show_progress, "Loading NEG/DON", sequence_length)

    all_data = []
    all_labels = []
    
    for data_arr, label_arr in [(acc_can_data, acc_can_labels), (acc_nc_data, acc_nc_labels), 
                                (don_can_data, don_can_labels), (don_nc_data, don_nc_labels)]:
        if len(data_arr) > 0:
            all_data.append(data_arr)
            all_labels.append(label_arr)
    
    for data_arr, label_arr in [(acc_syn_data, acc_syn_labels), (don_syn_data, don_syn_labels)]:
        if len(data_arr) > 0:
            all_data.append(data_arr)
            all_labels.append(label_arr)
    
    for data_arr, label_arr in [(neg_acc_data, neg_acc_labels), (neg_don_data, neg_don_labels)]:
        if len(data_arr) > 0:
            all_data.append(data_arr)
            all_labels.append(label_arr)

    data = np.concatenate(all_data)
    labels = np.concatenate(all_labels)

    print(f"[INFO] Loaded data shapes: {data.shape}, {labels.shape}")
    print(f"[INFO] Class distribution: {np.bincount(labels)}")
    print("[INFO] Class mapping: 0=ACC/CAN, 1=ACC/NC, 2=DON/CAN, 3=DON/NC, 4=Non-splice site")
    
    return data, labels

def load_data_seven_class_with_negatives(base_path, adasyn_subdir, show_progress=False):
    
    pos_path = os.path.join(base_path, 'POS')
    neg_path = os.path.join(base_path, 'NEG')
    acc_path = os.path.join(pos_path, 'ACC')
    don_path = os.path.join(pos_path, 'DON')

    print("[INFO] Loading ACC/CAN data...")
    acc_can_data, acc_can_labels = load_sequences_from_folder(os.path.join(acc_path, 'CAN'), 0, show_progress, "Loading ACC/CAN")
    
    print("[INFO] Loading ACC/NC data...")
    acc_nc_data, acc_nc_labels = load_sequences_from_folder(os.path.join(acc_path, 'NC'), 1, show_progress, "Loading ACC/NC")
    
    acc_syn_data, acc_syn_labels = np.array([]), np.array([])
    if adasyn_subdir:
        adasyn_acc_path = os.path.join(acc_path, 'ADASYN', adasyn_subdir)
        if os.path.exists(adasyn_acc_path):
            print(f"[INFO] Loading ACC/ADASYN/{adasyn_subdir} data (separate class)...")
            acc_syn_data, acc_syn_labels = load_sequences_from_folder(adasyn_acc_path, 2, show_progress, f"Loading ACC/ADASYN/{adasyn_subdir}")

    print("[INFO] Loading DON/CAN data...")
    don_can_data, don_can_labels = load_sequences_from_folder(os.path.join(don_path, 'CAN'), 3, show_progress, "Loading DON/CAN")
    
    print("[INFO] Loading DON/NC data...")
    don_nc_data, don_nc_labels = load_sequences_from_folder(os.path.join(don_path, 'NC'), 4, show_progress, "Loading DON/NC")
    
    don_syn_data, don_syn_labels = np.array([]), np.array([])
    if adasyn_subdir:
        adasyn_don_path = os.path.join(don_path, 'ADASYN', adasyn_subdir)
        if os.path.exists(adasyn_don_path):
            print(f"[INFO] Loading DON/ADASYN/{adasyn_subdir} data (separate class)...")
            don_syn_data, don_syn_labels = load_sequences_from_folder(adasyn_don_path, 5, show_progress, f"Loading DON/ADASYN/{adasyn_subdir}")

    print("[INFO] Loading NEG sequences (non-splice sites)...")
    neg_acc_data, neg_acc_labels = load_sequences_from_folder(os.path.join(neg_path, 'ACC'), 6, show_progress, "Loading NEG/ACC")
    neg_don_data, neg_don_labels = load_sequences_from_folder(os.path.join(neg_path, 'DON'), 6, show_progress, "Loading NEG/DON")

    all_data = []
    all_labels = []
    
    # Add all sequences
    for data_arr, label_arr in [(acc_can_data, acc_can_labels), (acc_nc_data, acc_nc_labels), 
                                (don_can_data, don_can_labels), (don_nc_data, don_nc_labels),
                                (neg_acc_data, neg_acc_labels), (neg_don_data, neg_don_labels)]:
        if len(data_arr) > 0:
            all_data.append(data_arr)
            all_labels.append(label_arr)
    
    # Add synthetic sequences if they exist
    for data_arr, label_arr in [(acc_syn_data, acc_syn_labels), (don_syn_data, don_syn_labels)]:
        if len(data_arr) > 0:
            all_data.append(data_arr)
            all_labels.append(label_arr)

    data = np.concatenate(all_data)
    labels = np.concatenate(all_labels)

    print(f"[INFO] Loaded data shapes: {data.shape}, {labels.shape}")
    print(f"[INFO] Class distribution: {np.bincount(labels)}")
    print("[INFO] Class mapping: 0=ACC/CAN, 1=ACC/NC, 2=ACC/ADASYN, 3=DON/CAN, 4=DON/NC, 5=DON/ADASYN, 6=Non-splice site")
    
    return data, labels

def load_data_three_class(base_path, adasyn_subdir=None, show_progress=False, sequence_length=600):
    pos_path = os.path.join(base_path, 'POS')
    neg_path = os.path.join(base_path, 'NEG')
    acc_path = os.path.join(pos_path, 'ACC')
    don_path = os.path.join(pos_path, 'DON')

    all_data = []
    all_labels = []

    if adasyn_subdir:
        print("[INFO] Loading Acceptor sequences (ACC/CAN + ACC/NC + ACC/ADASYN)...")
    else:
        print("[INFO] Loading Acceptor sequences (ACC/CAN + ACC/NC only)...")
    
    acc_can_data, acc_can_labels = load_sequences_from_folder(os.path.join(acc_path, 'CAN'), 0, show_progress, "Loading ACC/CAN", sequence_length)
    if len(acc_can_data) > 0:
        all_data.append(acc_can_data)
        all_labels.append(acc_can_labels)
    
    acc_nc_data, acc_nc_labels = load_sequences_from_folder(os.path.join(acc_path, 'NC'), 0, show_progress, "Loading ACC/NC", sequence_length)
    if len(acc_nc_data) > 0:
        all_data.append(acc_nc_data)
        all_labels.append(acc_nc_labels)
    
    if adasyn_subdir:
        adasyn_acc_path = os.path.join(acc_path, 'ADASYN', adasyn_subdir)
        if os.path.exists(adasyn_acc_path):
            print(f"[INFO] Loading ACC/ADASYN/{adasyn_subdir} (as Acceptor)...")
            acc_syn_data, acc_syn_labels = load_sequences_from_folder(adasyn_acc_path, 0, show_progress, f"Loading ACC/ADASYN/{adasyn_subdir}", sequence_length)
            if len(acc_syn_data) > 0:
                all_data.append(acc_syn_data)
                all_labels.append(acc_syn_labels)
        else:
            print(f"[WARN] ADASYN folder not found: {adasyn_acc_path}")

    # ==========  DONOR (Label 1) ==========
    if adasyn_subdir:
        print("[INFO] Loading Donor sequences (DON/CAN + DON/NC + DON/ADASYN)...")
    else:
        print("[INFO] Loading Donor sequences (DON/CAN + DON/NC only)...")
    
    # DON/CAN
    don_can_data, don_can_labels = load_sequences_from_folder(os.path.join(don_path, 'CAN'), 1, show_progress, "Loading DON/CAN", sequence_length)
    if len(don_can_data) > 0:
        all_data.append(don_can_data)
        all_labels.append(don_can_labels)
    
    # DON/NC
    don_nc_data, don_nc_labels = load_sequences_from_folder(os.path.join(don_path, 'NC'), 1, show_progress, "Loading DON/NC", sequence_length)
    if len(don_nc_data) > 0:
        all_data.append(don_nc_data)
        all_labels.append(don_nc_labels)
    
    # DON/ADASYN (if specified)
    if adasyn_subdir:
        adasyn_don_path = os.path.join(don_path, 'ADASYN', adasyn_subdir)
        if os.path.exists(adasyn_don_path):
            print(f"[INFO] Loading DON/ADASYN/{adasyn_subdir} (as Donor)...")
            don_syn_data, don_syn_labels = load_sequences_from_folder(adasyn_don_path, 1, show_progress, f"Loading DON/ADASYN/{adasyn_subdir}", sequence_length)
            if len(don_syn_data) > 0:
                all_data.append(don_syn_data)
                all_labels.append(don_syn_labels)
        else:
            print(f"[WARN] ADASYN folder not found: {adasyn_don_path}")

    # ==========  NO SPLICE SITE (Label 2) ==========
    print("[INFO] Loading No Splice Site sequences (NEG/ACC + NEG/DON)...")
    
    # NEG/ACC
    neg_acc_data, neg_acc_labels = load_sequences_from_folder(os.path.join(neg_path, 'ACC'), 2, show_progress, "Loading NEG/ACC", sequence_length)
    if len(neg_acc_data) > 0:
        all_data.append(neg_acc_data)
        all_labels.append(neg_acc_labels)
    
    # NEG/DON
    neg_don_data, neg_don_labels = load_sequences_from_folder(os.path.join(neg_path, 'DON'), 2, show_progress, "Loading NEG/DON", sequence_length)
    if len(neg_don_data) > 0:
        all_data.append(neg_don_data)
        all_labels.append(neg_don_labels)

    data = np.concatenate(all_data)
    labels = np.concatenate(all_labels)

    print(f"[INFO] Loaded data shapes: {data.shape}, {labels.shape}")
    print(f"[INFO] Class distribution: {np.bincount(labels)}")
    print("[INFO] Class mapping: 0=Acceptor (all ACC types), 1=Donor (all DON types), 2=No Splice Site (all NEG types)")
    
    return data, labels

def load_test_data_three_class(base_path, show_progress=False, sequence_length=600):
   
    return load_data_three_class(base_path, adasyn_subdir=None, show_progress=show_progress, sequence_length=sequence_length)

def load_data_from_folder(
        base_path, 
        include_neg=False, 
        include_synthetic=False, 
        adasyn_subfolder=None,    # Specify which ADASYN subfolder (e.g., "ADASYN_SNC_100")
        show_progress=False,
        use_six_class_labels=False  # Use 6-class labeling like load_data function
    ):
   
    
    if use_six_class_labels:
        if not adasyn_subfolder:
            raise ValueError("adasyn_subfolder must be specified when use_six_class_labels=True")
        return load_data(base_path, adasyn_subfolder, show_progress)
    
    # Original logic for 2-4 class labels
    X, y = [], []

    acc_dir = os.path.join(base_path, 'POS', 'ACC')
    if os.path.exists(acc_dir):
        for subtype in os.listdir(acc_dir):
            subtype_path = os.path.join(acc_dir, subtype)
            if not os.path.isdir(subtype_path):
                continue

            subtype_upper = subtype.upper()
            
            # Assign proper labels for ACC data
            if subtype_upper == "CAN":
                label = 0  # ACC/CAN
                files = os.listdir(subtype_path)
                iterator = tqdm(files, desc=f"Loading ACC/CAN", disable=not show_progress)
                for fname in iterator:
                    fpath = os.path.join(subtype_path, fname)
                    try:
                        with open(fpath, 'r') as f:
                            seq = f.read().strip()
                            if len(seq) == 600:
                                X.append(one_hot_encode(seq))
                                y.append(label)
                    except Exception as e:
                        print(f"[WARN] Skipping {fpath}: {e}")
                        
            elif subtype_upper == "NC":
                label = 1  # ACC/NC
                files = os.listdir(subtype_path)
                iterator = tqdm(files, desc=f"Loading ACC/NC", disable=not show_progress)
                for fname in iterator:
                    fpath = os.path.join(subtype_path, fname)
                    try:
                        with open(fpath, 'r') as f:
                            seq = f.read().strip()
                            if len(seq) == 600:
                                X.append(one_hot_encode(seq))
                                y.append(label)
                    except Exception as e:
                        print(f"[WARN] Skipping {fpath}: {e}")

            elif (include_synthetic and adasyn_subfolder and subtype_upper == "ADASYN"):
                adasyn_path = os.path.join(subtype_path, adasyn_subfolder)
                if os.path.isdir(adasyn_path):
                    label = 1  # ACC/ADASYN goes to ACC/NC after collapse
                    files = os.listdir(adasyn_path)
                    iterator = tqdm(files, desc=f"Loading ACC/ADASYN/{adasyn_subfolder}", disable=not show_progress)
                    for fname in iterator:
                        fpath = os.path.join(adasyn_path, fname)
                        try:
                            with open(fpath, 'r') as f:
                                seq = f.read().strip()
                                if len(seq) == 600:
                                    X.append(one_hot_encode(seq))
                                    y.append(label)
                        except Exception as e:
                            print(f"[WARN] Skipping {fpath}: {e}")
                else:
                    print(f"[WARN] ADASYN subfolder not found: {adasyn_path}")

    don_dir = os.path.join(base_path, 'POS', 'DON')
    if os.path.exists(don_dir):
        for subtype in os.listdir(don_dir):
            subtype_path = os.path.join(don_dir, subtype)
            if not os.path.isdir(subtype_path):
                continue

            subtype_upper = subtype.upper()
            
            if subtype_upper == "CAN":
                label = 3  # DON/CAN
                files = os.listdir(subtype_path)
                iterator = tqdm(files, desc=f"Loading DON/CAN", disable=not show_progress)
                for fname in iterator:
                    fpath = os.path.join(subtype_path, fname)
                    try:
                        with open(fpath, 'r') as f:
                            seq = f.read().strip()
                            if len(seq) == 600:
                                X.append(one_hot_encode(seq))
                                y.append(label)
                    except Exception as e:
                        print(f"[WARN] Skipping {fpath}: {e}")
                        
            elif subtype_upper == "NC":
                label = 4  # DON/NC
                files = os.listdir(subtype_path)
                iterator = tqdm(files, desc=f"Loading DON/NC", disable=not show_progress)
                for fname in iterator:
                    fpath = os.path.join(subtype_path, fname)
                    try:
                        with open(fpath, 'r') as f:
                            seq = f.read().strip()
                            if len(seq) == 600:
                                X.append(one_hot_encode(seq))
                                y.append(label)
                    except Exception as e:
                        print(f"[WARN] Skipping {fpath}: {e}")

            # Include only the specified ADASYN subfolder if requested
            elif (include_synthetic and adasyn_subfolder and subtype_upper == "ADASYN"):
                adasyn_path = os.path.join(subtype_path, adasyn_subfolder)
                if os.path.isdir(adasyn_path):
                    label = 4  # DON/ADASYN goes to DON/NC after collapse
                    files = os.listdir(adasyn_path)
                    iterator = tqdm(files, desc=f"Loading DON/ADASYN/{adasyn_subfolder}", disable=not show_progress)
                    for fname in iterator:
                        fpath = os.path.join(adasyn_path, fname)
                        try:
                            with open(fpath, 'r') as f:
                                seq = f.read().strip()
                                if len(seq) == 600:
                                    X.append(one_hot_encode(seq))
                                    y.append(label)
                        except Exception as e:
                            print(f"[WARN] Skipping {fpath}: {e}")
                else:
                    print(f"[WARN] ADASYN subfolder not found: {adasyn_path}")

    # NEGATIVE: ACC (2), DON (5) if requested  
    if include_neg:
        for class_name, label in [('ACC', 2), ('DON', 5)]:
            neg_path = os.path.join(base_path, 'NEG', class_name)
            if not os.path.exists(neg_path):
                print(f"[WARN] NEG folder not found: {neg_path}")
                continue
            files = os.listdir(neg_path)
            iterator = tqdm(files, desc=f"Loading NEG/{class_name}", disable=not show_progress)
            for fname in iterator:
                fpath = os.path.join(neg_path, fname)
                try:
                    with open(fpath, 'r') as f:
                        seq = f.read().strip()
                        if len(seq) == 600:
                            X.append(one_hot_encode(seq))
                            y.append(label)
                except Exception as e:
                    print(f"[WARN] Skipping {fpath}: {e}")

    print(f"[INFO] Loaded {len(X)} sequences from {base_path} (include_neg={include_neg}, include_synthetic={include_synthetic}, adasyn_subfolder={adasyn_subfolder})")
    return np.array(X), np.array(y)

def load_base_data_three_class(base_path, show_progress=False, sequence_length=600):
    pos_path = os.path.join(base_path, 'POS')
    neg_path = os.path.join(base_path, 'NEG')
    acc_path = os.path.join(pos_path, 'ACC')
    don_path = os.path.join(pos_path, 'DON')

    all_data = []
    all_labels = []

    # ==========  ACCEPTOR (Label 0) ==========
    print("[INFO] Loading base Acceptor sequences (ACC/CAN + ACC/NC only)...")
    
    # ACC/CAN
    acc_can_data, acc_can_labels = load_sequences_from_folder(os.path.join(acc_path, 'CAN'), 0, show_progress, "Loading ACC/CAN", sequence_length)
    if len(acc_can_data) > 0:
        all_data.append(acc_can_data)
        all_labels.append(acc_can_labels)
    
    # ACC/NC
    acc_nc_data, acc_nc_labels = load_sequences_from_folder(os.path.join(acc_path, 'NC'), 0, show_progress, "Loading ACC/NC", sequence_length)
    if len(acc_nc_data) > 0:
        all_data.append(acc_nc_data)
        all_labels.append(acc_nc_labels)

    # ==========  DONOR (Label 1) ==========
    print("[INFO] Loading base Donor sequences (DON/CAN + DON/NC only)...")
    
    # DON/CAN
    don_can_data, don_can_labels = load_sequences_from_folder(os.path.join(don_path, 'CAN'), 1, show_progress, "Loading DON/CAN", sequence_length)
    if len(don_can_data) > 0:
        all_data.append(don_can_data)
        all_labels.append(don_can_labels)
    
    # DON/NC
    don_nc_data, don_nc_labels = load_sequences_from_folder(os.path.join(don_path, 'NC'), 1, show_progress, "Loading DON/NC", sequence_length)
    if len(don_nc_data) > 0:
        all_data.append(don_nc_data)
        all_labels.append(don_nc_labels)

    # ==========  NO SPLICE SITE (Label 2) ==========
    print("[INFO] Loading No Splice Site sequences (NEG/ACC + NEG/DON)...")
    
    # NEG/ACC
    neg_acc_data, neg_acc_labels = load_sequences_from_folder(os.path.join(neg_path, 'ACC'), 2, show_progress, "Loading NEG/ACC", sequence_length)
    if len(neg_acc_data) > 0:
        all_data.append(neg_acc_data)
        all_labels.append(neg_acc_labels)
    
    # NEG/DON
    neg_don_data, neg_don_labels = load_sequences_from_folder(os.path.join(neg_path, 'DON'), 2, show_progress, "Loading NEG/DON", sequence_length)
    if len(neg_don_data) > 0:
        all_data.append(neg_don_data)
        all_labels.append(neg_don_labels)

    # Combine all data and labels
    data = np.concatenate(all_data)
    labels = np.concatenate(all_labels)

    print(f"[INFO] Loaded base data shapes: {data.shape}, {labels.shape}")
    print(f"[INFO] Base class distribution: {np.bincount(labels)}")
    
    return data, labels

def load_base_data_three_class_separated(train_dir, show_progress=False, sequence_length=600):
    print("[INFO] Loading base 3-class data with canonical/non-canonical separation...")
    
    acc_can_path = os.path.join(train_dir, "POS", "ACC", "CAN")
    acc_can_data, acc_can_labels = load_sequences_from_folder(
        acc_can_path, 0, show_progress, "Loading ACC/CAN", sequence_length
    )
    
    acc_nc_path = os.path.join(train_dir, "POS", "ACC", "NC")
    acc_nc_data, acc_nc_labels = load_sequences_from_folder(
        acc_nc_path, 0, show_progress, "Loading ACC/NC", sequence_length
    )
    
    don_can_path = os.path.join(train_dir, "POS", "DON", "CAN")
    don_can_data, don_can_labels = load_sequences_from_folder(
        don_can_path, 1, show_progress, "Loading DON/CAN", sequence_length
    )
    
    don_nc_path = os.path.join(train_dir, "POS", "DON", "NC")
    don_nc_data, don_nc_labels = load_sequences_from_folder(
        don_nc_path, 1, show_progress, "Loading DON/NC", sequence_length
    )
    
    neg_acc_path = os.path.join(train_dir, "NEG", "ACC")
    neg_acc_data, neg_acc_labels = load_sequences_from_folder(
        neg_acc_path, 2, show_progress, "Loading NEG/ACC", sequence_length
    ) if os.path.exists(neg_acc_path) else (np.array([]), np.array([]))
    
    neg_don_path = os.path.join(train_dir, "NEG", "DON")
    neg_don_data, neg_don_labels = load_sequences_from_folder(
        neg_don_path, 2, show_progress, "Loading NEG/DON", sequence_length
    ) if os.path.exists(neg_don_path) else (np.array([]), np.array([]))
    
    if len(neg_acc_data) > 0 and len(neg_don_data) > 0:
        neg_data = np.concatenate([neg_acc_data, neg_don_data])
        neg_labels = np.concatenate([neg_acc_labels, neg_don_labels])
    elif len(neg_acc_data) > 0:
        neg_data = neg_acc_data
        neg_labels = neg_acc_labels
    elif len(neg_don_data) > 0:
        neg_data = neg_don_data
        neg_labels = neg_don_labels
    else:
        neg_data = np.array([])
        neg_labels = np.array([])
    
    print(f"[INFO] Separated data loaded:")
    print(f"  ACC/CAN: {len(acc_can_data)} sequences (label 0)")
    print(f"  ACC/NC: {len(acc_nc_data)} sequences (label 0)")  
    print(f"  DON/CAN: {len(don_can_data)} sequences (label 1)")
    print(f"  DON/NC: {len(don_nc_data)} sequences (label 1)")
    print(f"  NEG: {len(neg_data)} sequences (label 2)")
    
    total_expected = len(acc_can_data) + len(acc_nc_data) + len(don_can_data) + len(don_nc_data) + len(neg_data)
    print(f"[INFO] Total sequences: {total_expected}")
    
    return (
        (acc_can_data, acc_can_labels),
        (acc_nc_data, acc_nc_labels), 
        (don_can_data, don_can_labels),
        (don_nc_data, don_nc_labels),
        (neg_data, neg_labels)
    )

def load_synthetic_data_three_class(base_path, adasyn_subdir, show_progress=False):
    pos_path = os.path.join(base_path, 'POS')
    acc_path = os.path.join(pos_path, 'ACC')
    don_path = os.path.join(pos_path, 'DON')

    all_data = []
    all_labels = []

    print(f"[INFO] Loading synthetic data from {adasyn_subdir}...")

    # ACC/ADASYN
    adasyn_acc_path = os.path.join(acc_path, 'ADASYN', adasyn_subdir)
    if os.path.exists(adasyn_acc_path):
        print(f"[INFO] Loading ACC/ADASYN/{adasyn_subdir} (as Acceptor)...")
        acc_syn_data, acc_syn_labels = load_sequences_from_folder(adasyn_acc_path, 0, show_progress, f"Loading ACC/ADASYN/{adasyn_subdir}")
        if len(acc_syn_data) > 0:
            all_data.append(acc_syn_data)
            all_labels.append(acc_syn_labels)
    else:
        print(f"[WARN] ADASYN folder not found: {adasyn_acc_path}")

    # DON/ADASYN
    adasyn_don_path = os.path.join(don_path, 'ADASYN', adasyn_subdir)
    if os.path.exists(adasyn_don_path):
        print(f"[INFO] Loading DON/ADASYN/{adasyn_subdir} (as Donor)...")
        don_syn_data, don_syn_labels = load_sequences_from_folder(adasyn_don_path, 1, show_progress, f"Loading DON/ADASYN/{adasyn_subdir}")
        if len(don_syn_data) > 0:
            all_data.append(don_syn_data)
            all_labels.append(don_syn_labels)
    else:
        print(f"[WARN] ADASYN folder not found: {adasyn_don_path}")

    if len(all_data) == 0:
        print("[WARN] No synthetic data found!")
        return np.array([]).reshape(0, 600, 4), np.array([])

    data = np.concatenate(all_data)
    labels = np.concatenate(all_labels)

    print(f"[INFO] Loaded synthetic data shapes: {data.shape}, {labels.shape}")
    print(f"[INFO] Synthetic class distribution: {np.bincount(labels)}")
    
    return data, labels