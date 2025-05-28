import os
import pandas as pd
import matplotlib.pyplot as plt

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
                gc, at = calculate_nucleotide_content(sequence)
                data.append([gc, at, label])
    return data

def plot_data(df, title, labels, markers, colors, save_path=None):
    plt.figure(figsize=(8, 6))
    for label, marker, color, desc in zip(labels['values'], markers, colors, labels['desc']):
        subset = df[df['Label'] == label]
        plt.scatter(subset['GC_Content'], subset['AT_Content'], 
                    label=desc, marker=marker, color=color, alpha=0.7, edgecolor='k')
    plt.xlabel('GC Content')
    plt.ylabel('AT Content')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.5)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    else:
        plt.show()
    plt.close()

def create_plots(canonical_folder, noncanonical_folder, synthetic_folder, title_prefix):
    can_data = process_sequences(canonical_folder, label=0)
    nc_data = process_sequences(noncanonical_folder, label=1)
    syn_data = process_sequences(synthetic_folder, label=2)

    df_original = pd.DataFrame(can_data + nc_data, columns=['GC_Content', 'AT_Content', 'Label'])
    df_combined = pd.DataFrame(can_data + nc_data + syn_data, columns=['GC_Content', 'AT_Content', 'Label'])
    df_combined['Label'] = df_combined['Label'].replace({2: 1})  # Combine NC + synthetic

    plot_data(
        df_original,
        title=f"{title_prefix} - Canonical vs Non-Canonical",
        labels={'values': [0, 1], 'desc': ['Canonical', 'Non-Canonical']},
        markers=['o', 's'],
        colors=['blue', 'red'],
        save_path=f"{title_prefix.replace(' ', '_')}_Canonical_vs_Non_Canonical.png"
    )

    plot_data(
        df_combined,
        title=f"{title_prefix} - Canonical vs Combined Non-Canonical",
        labels={'values': [0, 1], 'desc': ['Canonical', 'Non-Canonical + Synthetic']},
        markers=['o', '^'],
        colors=['blue', 'green'],
        save_path=f"{title_prefix.replace(' ', '_')}_Canonical_vs_Combined_Non_Canonical.png"
    )
