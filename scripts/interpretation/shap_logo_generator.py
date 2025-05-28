import os
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Conv1D, BatchNormalization, Add, Activation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logomaker

# === Custom Residual Block ===
class ResidualBlock(Layer):
    def __init__(self, filters, kernel_size, strides=1, use_activation=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_activation = use_activation
        self.conv1 = Conv1D(filters, kernel_size, strides=strides, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv1D(filters, kernel_size, strides=1, padding='same', use_bias=False)
        self.bn2 = BatchNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        if self.use_activation:
            x = Activation('relu')(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return Add()([inputs, x])

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'use_activation': self.use_activation
        })
        return config

# === One-hot Encoding ===
NUCLEOTIDE_MAP = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}

def one_hot_encode(sequence):
    return np.array([NUCLEOTIDE_MAP.get(nt, [0, 0, 0, 0]) for nt in sequence])

# === Data Loading ===
def load_sequences_from_folder(folder_path, label):
    data, labels = [], []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 600:
                    data.append(one_hot_encode(line))
                    labels.append(label)
    return np.array(data), np.array(labels)

def load_data(base_path):
    data, labels = [], []
    if os.path.isdir(os.path.join(base_path, 'CAN')) or os.path.isdir(os.path.join(base_path, 'NC')):
        if os.path.exists(os.path.join(base_path, 'CAN')):
            d, l = load_sequences_from_folder(os.path.join(base_path, 'CAN'), 0)
            data.append(d)
            labels.append(l)
        if os.path.exists(os.path.join(base_path, 'NC')):
            d, l = load_sequences_from_folder(os.path.join(base_path, 'NC'), 1)
            data.append(d)
            labels.append(l)
    else:
        acc_path = os.path.join(base_path, 'ACC')
        don_path = os.path.join(base_path, 'DON')
        if os.path.exists(os.path.join(acc_path, 'CAN')):
            d, l = load_sequences_from_folder(os.path.join(acc_path, 'CAN'), 0)
            data.append(d)
            labels.append(l)
        if os.path.exists(os.path.join(acc_path, 'NC')):
            d, l = load_sequences_from_folder(os.path.join(acc_path, 'NC'), 1)
            data.append(d)
            labels.append(l)
        if os.path.exists(os.path.join(don_path, 'CAN')):
            d, l = load_sequences_from_folder(os.path.join(don_path, 'CAN'), 2)
            data.append(d)
            labels.append(l)
        if os.path.exists(os.path.join(don_path, 'NC')):
            d, l = load_sequences_from_folder(os.path.join(don_path, 'NC'), 3)
            data.append(d)
            labels.append(l)

    if not data:
        raise ValueError(f"No valid data found in: {base_path}")

    return np.concatenate(data), np.concatenate(labels)

# === SHAP-Weighted Sequence Logo Generation ===
def run_shap_weighted_logomaker(model_path, data_path, n_samples=100, class_index=1, output="shap_weighted_logo.png"):
    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(model_path, custom_objects={'ResidualBlock': ResidualBlock}, compile=False)

    print("[INFO] Loading data...")
    data, _ = load_data(data_path)
    if n_samples > len(data):
        n_samples = len(data)
    X_sample = data[:n_samples]
    background = data[np.random.choice(data.shape[0], size=min(50, len(data)), replace=False)]

    print("[INFO] Running SHAP...")
    try:
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        print(f"[WARN] GradientExplainer failed: {e}")
        print("[INFO] Falling back to KernelExplainer...")
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_sample)

    class_shap = shap_values[class_index]  # shape: (n_samples, 600, 4)
    n_samples, seq_len, _ = class_shap.shape

    print("[INFO] Computing SHAP-weighted nucleotide matrix...")
    pwm = np.zeros((seq_len, 4))  # shape: (600, 4)

    for i in range(n_samples):
        for pos in range(seq_len):
            base_index = np.argmax(X_sample[i, pos])  # actual base at that position
            shap_val = class_shap[i, pos, base_index]
            if np.isfinite(shap_val):
                pwm[pos, base_index] += shap_val

    # Restrict to position range 290-310
    start_pos, end_pos = 290, 311
    pwm = pwm[start_pos:end_pos]

    # Remove negatives and normalize
    pwm = np.maximum(pwm, 0)
    row_sums = pwm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    pwm = pwm / row_sums

    df_pwm = pd.DataFrame(pwm, columns=['A', 'C', 'G', 'T'])
    df_pwm.index = list(range(start_pos, end_pos))

    print("[INFO] Plotting logomaker logo...")
    fig, ax = plt.subplots(figsize=(12, 4))
    logomaker.Logo(df_pwm, ax=ax, color_scheme='classic')
    ax.set_title("SHAP-Weighted Sequence Logo (Positions 290-310)", fontsize=16)
    ax.set_xticks(list(range(start_pos, end_pos)))
    ax.set_xlabel("Position")
    ax.set_ylabel("Importance (Normalized)")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"[DONE] Logo saved to: {output}")
