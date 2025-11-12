# shap_logo_generator.py
import os
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, BatchNormalization, Add, Activation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logomaker

# ========= Residual Block with projection skip =========
class ResidualBlock(Layer):
    def __init__(self, filters, kernel_size, strides=1, use_activation=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_activation = use_activation

        self.conv1 = Conv1D(filters, kernel_size, strides=strides, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.act = Activation('relu')
        self.conv2 = Conv1D(filters, kernel_size, strides=1, padding='same', use_bias=False)
        self.bn2 = BatchNormalization()

        self.proj = None  # created in build if needed

    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        if in_ch != self.filters or self.strides != 1:
            self.proj = tf.keras.Sequential([
                Conv1D(self.filters, 1, strides=self.strides, padding='same', use_bias=False),
                BatchNormalization()
            ])
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        if self.use_activation:
            x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        shortcut = self.proj(inputs) if self.proj is not None else inputs
        out = Add()([shortcut, x])
        return self.act(out) if self.use_activation else out

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'use_activation': self.use_activation
        })
        return config

# ========= One-hot =========
NUCLEOTIDE_MAP = {'A': [1, 0, 0, 0],
                  'C': [0, 1, 0, 0],
                  'G': [0, 0, 1, 0],
                  'T': [0, 0, 0, 1]}

def one_hot_encode(sequence: str) -> np.ndarray:
    # unknowns → zeros; return float32 for TF/SHAP
    return np.array([NUCLEOTIDE_MAP.get(nt, [0, 0, 0, 0]) for nt in sequence], dtype=np.float32)

# ========= FASTA / line-per-seq helpers =========
def _read_sequences_from_file(file_path: str) -> list[str]:
    """
    Supports either:
      - one sequence per line (no headers), or
      - FASTA (headers start with '>')
    Returns a list of raw sequence strings (no whitespace).
    """
    seqs = []
    with open(file_path, 'r') as f:
        first = f.readline()
        if not first:
            return seqs
        is_fasta = first.startswith('>')
        f.seek(0)

        if not is_fasta:
            for line in f:
                s = line.strip().upper()
                if s:
                    seqs.append(s)
        else:
            curr = []
            for line in f:
                if line.startswith('>'):
                    if curr:
                        seqs.append(''.join(curr).upper())
                        curr = []
                else:
                    curr.append(line.strip())
            if curr:
                seqs.append(''.join(curr).upper())
    return seqs

def load_sequences_from_folder(folder_path: str, label: int, seq_len: int = 600):
    data, labels = [], []
    if not os.path.isdir(folder_path):
        return np.empty((0, seq_len, 4), dtype=np.float32), np.empty((0,), dtype=np.int32)
    for file_name in os.listdir(folder_path):
        fp = os.path.join(folder_path, file_name)
        if not os.path.isfile(fp):
            continue
        for seq in _read_sequences_from_file(fp):
            if len(seq) == seq_len:
                data.append(one_hot_encode(seq))
                labels.append(label)
    if not data:
        return np.empty((0, seq_len, 4), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.stack(data, axis=0), np.array(labels, dtype=np.int32)

def load_data(base_path: str, seq_len: int = 600):
    """
    Layouts supported:
      - base/CAN, base/NC  (binary)
      - base/ACC/{CAN,NC}, base/DON/{CAN,NC} (4-class)
    """
    data, labels = [], []

    # binary layout
    if os.path.isdir(os.path.join(base_path, 'CAN')) or os.path.isdir(os.path.join(base_path, 'NC')):
        d, l = load_sequences_from_folder(os.path.join(base_path, 'CAN'), 0, seq_len)
        if len(d): data.append(d); labels.append(l)
        d, l = load_sequences_from_folder(os.path.join(base_path, 'NC'), 1, seq_len)
        if len(d): data.append(d); labels.append(l)
    else:
        # 4-class (ACC/DON × CAN/NC)
        acc_path = os.path.join(base_path, 'ACC')
        don_path = os.path.join(base_path, 'DON')
        d, l = load_sequences_from_folder(os.path.join(acc_path, 'CAN'), 0, seq_len)  # keep your label map if needed
        if len(d): data.append(d); labels.append(l)
        d, l = load_sequences_from_folder(os.path.join(acc_path, 'NC'), 1, seq_len)
        if len(d): data.append(d); labels.append(l)
        d, l = load_sequences_from_folder(os.path.join(don_path, 'CAN'), 2, seq_len)
        if len(d): data.append(d); labels.append(l)
        d, l = load_sequences_from_folder(os.path.join(don_path, 'NC'), 3, seq_len)
        if len(d): data.append(d); labels.append(l)

    if not data:
        raise ValueError(f"No valid data found in: {base_path}")

    return np.concatenate(data, axis=0), np.concatenate(labels, axis=0)

# ========= SHAP-Weighted Logo =========
def run_shap_weighted_logomaker(
    model_path,
    data_path,
    n_samples=100,
    class_index=1,
    output="shap_weighted_logo.png",
):
    rng = np.random.default_rng(42)

    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'ResidualBlock': ResidualBlock},
        compile=False
    )

    print("[INFO] Loading data...")
    data, _ = load_data(data_path, seq_len=600)
    if len(data) == 0:
        raise ValueError("Loaded dataset is empty after filtering for length 600.")

    n_samples = min(int(n_samples), len(data))
    X_sample = data[:n_samples].astype(np.float32)

    bg_size = min(64, len(data))
    bg_idx = rng.choice(len(data), size=bg_size, replace=False)
    background = data[bg_idx].astype(np.float32)

    print("[INFO] Running SHAP...")
    # Prefer DeepExplainer; fall back to GradientExplainer; if both fail (common on TF>=2.4)
    # use Integrated Gradients as a robust alternative producing input-shape attributions.
    shap_values = None
    try:
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        print(f"[WARN] DeepExplainer failed: {e}")
        print("[INFO] Trying GradientExplainer...")
        try:
            explainer = shap.GradientExplainer(model, background)
            shap_values = explainer.shap_values(X_sample)
        except Exception as e2:
            print(f"[WARN] GradientExplainer failed: {e2}")
            print("[INFO] Falling back to Integrated Gradients...")

    if shap_values is None:
        # -------- Integrated Gradients fallback --------
        # Compute IG for the specified class_index; returns (N, L, 4)
        def select_class_output(preds: tf.Tensor) -> tf.Tensor:
            # preds shape: (N,) or (N,C)
            if preds.shape.rank is None:
                # fallback: treat as (N,C)
                c = int(class_index)
                return preds[..., c]
            if preds.shape.rank == 1:
                return preds
            # rank==2: pick class column
            c = int(class_index)
            return preds[:, c]

        def integrated_gradients(inputs: np.ndarray, steps: int = 64) -> np.ndarray:
            baseline = np.zeros_like(inputs, dtype=np.float32)
            inputs_tf = tf.convert_to_tensor(inputs, dtype=tf.float32)
            baseline_tf = tf.convert_to_tensor(baseline, dtype=tf.float32)
            total_grads = tf.zeros_like(inputs_tf)

            for k in range(1, int(steps) + 1):
                alpha = tf.cast(k, tf.float32) / tf.cast(steps, tf.float32)
                x = baseline_tf + alpha * (inputs_tf - baseline_tf)
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    y = model(x, training=False)
                    preds = select_class_output(y)
                grads = tape.gradient(preds, x)
                grads = tf.where(tf.math.is_finite(grads), grads, tf.zeros_like(grads))
                total_grads += grads

            avg_grads = total_grads / tf.cast(steps, tf.float32)
            ig = (inputs_tf - baseline_tf) * avg_grads
            return ig.numpy()

        class_shap = integrated_gradients(X_sample, steps=64)
    else:
        # Handle binary vs multiclass: SHAP may return an array or a list of arrays
        if isinstance(shap_values, list):
            if not (0 <= class_index < len(shap_values)):
                raise ValueError(f"class_index {class_index} out of range for {len(shap_values)} outputs.")
            class_shap = shap_values[class_index]
        else:
            class_shap = shap_values  # single output

    if class_shap.shape[:2] != X_sample.shape[:2]:
        raise ValueError(
            f"Shape mismatch: SHAP {class_shap.shape} vs inputs {X_sample.shape} on first two dims."
        )

    n_s, seq_len, num_ch = class_shap.shape
    if num_ch != 4:
        raise ValueError(f"Expected 4 channels (A,C,G,T), got {num_ch}.")

    print("[INFO] Computing SHAP-weighted nucleotide matrix...")
    pwm = np.zeros((seq_len, 4), dtype=np.float64)
    # accumulate SHAP only for the observed base at each position
    base_idx = np.argmax(X_sample, axis=-1)  # (n_s, seq_len)
    for i in range(n_s):
        rows = np.arange(seq_len)
        cols = base_idx[i]  # (seq_len,)
        vals = class_shap[i, rows, cols]
        vals = np.where(np.isfinite(vals), vals, 0.0)
        pwm[rows, cols] += vals

    # Window (positions 290–310 inclusive of 310)
    start_pos, end_pos = 290, 311
    pwm = pwm[start_pos:end_pos]           # (21, 4)

    # Remove negatives (keep only positive contribution) and row-normalize
    pwm = np.maximum(pwm, 0.0)
    row_sums = pwm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    pwm = pwm / row_sums

    df_pwm = pd.DataFrame(pwm, columns=['A', 'C', 'G', 'T'])
    df_pwm.index = list(range(start_pos, end_pos))

    print("[INFO] Plotting logomaker logo...")
    fig, ax = plt.subplots(figsize=(12, 4))
    logomaker.Logo(df_pwm, ax=ax, color_scheme='classic')
    ax.set_title("SHAP-Weighted Sequence Logo (Positions 290–310)", fontsize=16)
    ax.set_xticks(list(range(start_pos, end_pos)))
    ax.set_xlabel("Position")
    ax.set_ylabel("Normalized importance")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"[DONE] Logo saved to: {output}")
