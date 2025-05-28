import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN

def calculate_nucleotide_content(sequence):
    """Compute GC and AT content of a DNA sequence."""
    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence) if len(sequence) > 0 else 0
    at_content = (sequence.count('A') + sequence.count('T')) / len(sequence) if len(sequence) > 0 else 0
    return gc_content, at_content

def extract_gc_at_features(folder_path, label):
    """Extract GC/AT features from sequences in a folder."""
    data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), 'r') as file:
                sequence = file.read().strip()
                gc, at = calculate_nucleotide_content(sequence)
                data.append([gc, at, label])
    print(f"Processed {len(data)} sequences from {folder_path}")
    return data

def adasyn_on_content_features(canonical_data, noncanonical_data, target_count):
    """Apply ADASYN based on GC/AT content features."""
    combined_data = canonical_data + noncanonical_data
    df = pd.DataFrame(combined_data, columns=["GC_Content", "AT_Content", "Label"])
    X = df[["GC_Content", "AT_Content"]].values
    y = df["Label"].values

    oversample = ADASYN(sampling_strategy={1: target_count}, random_state=42)
    X_resampled, y_resampled = oversample.fit_resample(X, y)

    synthetic_df = pd.DataFrame(X_resampled, columns=["GC_Content", "AT_Content"])
    synthetic_df["Label"] = y_resampled

    # Keep only synthetic non-canonical samples
    synthetic_samples = synthetic_df[synthetic_df["Label"] == 1].iloc[len(noncanonical_data):]
    print(f"Generated {len(synthetic_samples)} synthetic samples")

    return synthetic_samples[["GC_Content", "AT_Content"]].values.tolist()

def save_feature_samples(samples, output_folder, prefix="synthetic"):
    os.makedirs(output_folder, exist_ok=True)
    for i, (gc, at) in enumerate(samples):
        with open(os.path.join(output_folder, f"{prefix}_{i+1}.txt"), 'w') as f:
            f.write(f"GC_Content: {gc}, AT_Content: {at}")
    print(f"Saved {len(samples)} samples to {output_folder}")
