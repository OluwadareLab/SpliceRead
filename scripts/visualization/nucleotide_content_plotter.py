import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import where

def calculate_nucleotide_content(sequence):
    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
    at_content = (sequence.count('A') + sequence.count('T')) / len(sequence)
    return gc_content, at_content

def process_sequences(folder_path, label):
    data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), 'r') as file:
                sequence = file.read().strip()
                gc_content, at_content = calculate_nucleotide_content(sequence)
                data.append([gc_content, at_content, label])
    return data

def add_jitter(data, scale=0.1):
    return data + np.random.uniform(-scale, scale, size=data.shape)

def plot_scatter(X, y, title, legend_labels, save_path, colors):
    plt.figure(figsize=(8, 6))
    X_jittered = np.copy(X)
    X_jittered[:, 0] = add_jitter(X[:, 0])
    X_jittered[:, 1] = add_jitter(X[:, 1])
    for label, legend in legend_labels.items():
        row_ix = where(y == label)[0]
        plt.scatter(X_jittered[row_ix, 0], X_jittered[row_ix, 1],
                    label=legend, color=colors[label], alpha=0.7, edgecolor='k')
    plt.xlabel("GC Content")
    plt.ylabel("AT Content")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Plot saved as: {save_path}")

def generate_plots(canonical_folder, noncanonical_folder, synthetic_folder, title_prefix, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    canonical_data = process_sequences(canonical_folder, label=0)
    noncanonical_data = process_sequences(noncanonical_folder, label=1)
    synthetic_data = process_sequences(synthetic_folder, label=2)
    df_original = pd.DataFrame(canonical_data + noncanonical_data, columns=["GC_Content", "AT_Content", "Label"])
    X_original = df_original[["GC_Content", "AT_Content"]].values
    y_original = df_original["Label"].values
    plot_scatter(
        X_original, y_original,
        title=f"{title_prefix} - Canonical vs Non-Canonical",
        legend_labels={0: "Canonical", 1: "Non-Canonical"},
        save_path=os.path.join(save_directory, f"{title_prefix.replace(' ', '_')}_Canonical_vs_Non_Canonical.png"),
        colors={0: 'blue', 1: 'orange'}
    )
    df_combined = pd.DataFrame(canonical_data + noncanonical_data + synthetic_data, columns=["GC_Content", "AT_Content", "Label"])
    X_combined = df_combined[["GC_Content", "AT_Content"]].values
    y_combined = df_combined["Label"].values
    plot_scatter(
        X_combined, y_combined,
        title=f"{title_prefix} - Canonical vs Non-Canonical vs Synthetic",
        legend_labels={0: "Canonical", 1: "Non-Canonical", 2: "Synthetic Non-Canonical"},
        save_path=os.path.join(save_directory, f"{title_prefix.replace(' ', '_')}_Canonical_vs_Non_Canonical_vs_Synthetic.png"),
        colors={0: 'blue', 1: 'orange', 2: 'green'}
    )