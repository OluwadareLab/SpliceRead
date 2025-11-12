

import os
import argparse
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, BatchNormalization, Add, Activation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logomaker


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
        self.proj = None

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
        return {
            **super().get_config(),
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'use_activation': self.use_activation,
        }


NUCS = ['A','C','G','T']
NUC2V = {'A':[1,0,0,0], 'C':[0,1,0,0], 'G':[0,0,1,0], 'T':[0,0,0,1]}
def one_hot(seq: str) -> np.ndarray:
    return np.asarray([NUC2V.get(b, [0,0,0,0]) for b in seq], dtype=np.float32)


def _read_sequences_from_file(fp: str) -> list[str]:
    seqs = []
    with open(fp, 'r') as f:
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
            buf = []
            for line in f:
                if line.startswith('>'):
                    if buf:
                        seqs.append(''.join(buf).upper()); buf = []
                else:
                    buf.append(line.strip())
            if buf:
                seqs.append(''.join(buf).upper())
    return seqs

def load_sequences_from_folder(folder_path: str, seq_len: int) -> np.ndarray:
    xs = []
    if not os.path.isdir(folder_path):
        return np.empty((0, seq_len, 4), dtype=np.float32)
    for name in os.listdir(folder_path):
        fp = os.path.join(folder_path, name)
        if not os.path.isfile(fp): continue
        for s in _read_sequences_from_file(fp):
            if len(s) == seq_len:
                xs.append(one_hot(s))
    if not xs:
        return np.empty((0, seq_len, 4), dtype=np.float32)
    return np.stack(xs, axis=0)

def load_data(base_path: str, seq_len: int = 600) -> np.ndarray:
    Xs = []
    if os.path.isdir(os.path.join(base_path, 'CAN')) or os.path.isdir(os.path.join(base_path, 'NC')):
        Xs.append(load_sequences_from_folder(os.path.join(base_path, 'CAN'), seq_len))
        Xs.append(load_sequences_from_folder(os.path.join(base_path, 'NC'), seq_len))
    else:
        for sub in ['ACC', 'DON']:
            for cls in ['CAN', 'NC']:
                Xs.append(load_sequences_from_folder(os.path.join(base_path, sub, cls), seq_len))
    Xs = [x for x in Xs if x.size]
    if not Xs:
        raise ValueError(f"No sequences of length {seq_len} found under {base_path}")
    return np.concatenate(Xs, axis=0)


def model_to_logits(model: tf.keras.Model) -> tf.keras.Model:
    last = model.layers[-1]
    
    if isinstance(last, tf.keras.layers.Activation) and last.activation in (tf.keras.activations.sigmoid,
                                                                           tf.keras.activations.softmax):
        return tf.keras.Model(model.inputs, last.input)
    
    act = getattr(last, "activation", None)
    if callable(act) and act in (tf.keras.activations.sigmoid, tf.keras.activations.softmax):
        try:
            return tf.keras.Model(model.inputs, last.input)
        except Exception:
            pass
    return model


def select_class_output(preds: tf.Tensor, class_index: int) -> tf.Tensor:
    
    if preds.shape.rank == 1:
        return preds
    if preds.shape.rank == 2:
        C = preds.shape[-1]
        if C is None:
            C = tf.shape(preds)[-1]
        if isinstance(C, int) and C == 1:
            return tf.squeeze(preds, axis=-1)
        idx = tf.minimum(tf.cast(class_index, tf.int32), tf.shape(preds)[-1] - 1)
        return preds[:, idx]
    raise ValueError(f"Unexpected prediction shape: {preds.shape}")


def plot_signed_cwm(cwm_window: np.ndarray, start: int, out_png: str, autoscale=True):
    arr = cwm_window.copy()
    if autoscale and arr.size:
        amax = float(np.nanmax(np.abs(arr)))
        if 0 < amax < 1e-3:
            arr *= 1e3
            print("[INFO] Scaled CWM by 1e3× for visualization (values were tiny).")
    df = pd.DataFrame(arr, columns=NUCS, index=list(range(start, start + arr.shape[0])))
    fig, ax = plt.subplots(figsize=(12, 4))
    logomaker.Logo(df, ax=ax)
    ax.axhline(0, color='k', lw=1)
    ax.set_title(f"Signed SHAP Contribution Logo (Positions {start}–{start + arr.shape[0]-1})", fontsize=16)
    ax.set_xlabel("Position"); ax.set_ylabel("Mean SHAP contribution (signed)")
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

def plot_pwm(pwm_window: np.ndarray, start: int, out_png: str, info_content: bool = False, bg: float = 0.25):
    df = pd.DataFrame(pwm_window, columns=NUCS, index=list(range(start, start + pwm_window.shape[0])))
    fig, ax = plt.subplots(figsize=(12, 4))
    if info_content:
        q = np.array([bg, bg, bg, bg])[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(df.values, q, out=np.ones_like(df.values), where=df.values > 0)
            info = (df.values * (np.log2(ratio)))
        total_info = info.sum(axis=1, keepdims=True)
        total_info[total_info == 0] = 1.0
        scaled = np.divide(info, total_info, out=np.zeros_like(info), where=total_info > 0) * total_info
        df_plot = pd.DataFrame(scaled, columns=NUCS, index=df.index)
        logomaker.Logo(df_plot, ax=ax)
        ax.set_ylabel("Information (bits)")
        ax.set_title(f"PWM Information Logo (Positions {start}–{start + pwm_window.shape[0]-1})", fontsize=16)
    else:
        logomaker.Logo(df, ax=ax)
        ax.set_ylabel("Frequency")
        ax.set_title(f"PWM Frequency Logo (Positions {start}–{start + pwm_window.shape[0]-1})", fontsize=16)
    ax.set_xlabel("Position")
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()


def main():
    p = argparse.ArgumentParser(description="Generate signed SHAP CWM and true PWM logos.")
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--samples', type=int, default=256)
    p.add_argument('--class_index', type=int, default=0)
    p.add_argument('--seq_len', type=int, default=600)
    p.add_argument('--start', type=int, default=290)
    p.add_argument('--end', type=int, default=310)
    p.add_argument('--cwm_png', default="cwm_signed_logo.png")
    p.add_argument('--pwm_png', default="pwm_logo.png")
    p.add_argument('--pwm_info_png', default="pwm_info_logo.png")
    p.add_argument('--bg_size', type=int, default=200)
    p.add_argument('--ig_baseline', choices=['zero', 'mean', 'uniform'], default='mean',
                   help="Baseline for IG fallback: all-zeros, dataset mean, or uniform 0.25/base.")
    args = p.parse_args()

    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(args.model, custom_objects={'ResidualBlock': ResidualBlock}, compile=False)
    logits_model = model_to_logits(model)
    print("[DEBUG] logits_model output shape:", logits_model.output.shape)

    print("[INFO] Loading data...")
    Xall = load_data(args.data, seq_len=args.seq_len)
    if len(Xall) < 2:
        raise ValueError("Need at least 2 sequences.")
    rng = np.random.default_rng(42)
    n_samples = min(args.samples, len(Xall))
    idx = rng.choice(len(Xall), size=n_samples, replace=False)
    X = Xall[idx].astype(np.float32)

    
    prob = model(X[:8], training=False).numpy()
    logit = logits_model(X[:8], training=False).numpy()
    print("[DEBUG] prob mean/std:", float(prob.mean()), float(prob.std()))
    print("[DEBUG] logit mean/std:", float(logit.mean()), float(logit.std()))

    
    X_probe = tf.convert_to_tensor(X[:8], dtype=tf.float32)
    with tf.GradientTape() as t:
        t.watch(X_probe)
        ytest = logits_model(X_probe, training=False)
        scalar = select_class_output(ytest, args.class_index)
    gprobe = t.gradient(scalar, X_probe)
    gnorm = float(tf.linalg.global_norm([gprobe]))
    print("[DEBUG] grad L2-norm on mini-batch:", gnorm)

    
    base_idx = np.argmax(X, axis=-1)
    pfm_counts = np.zeros((args.seq_len, 4), dtype=np.float64)
    for i in range(n_samples):
        pfm_counts[np.arange(args.seq_len), base_idx[i]] += 1
    
    pfm = (pfm_counts + 0.5) / (pfm_counts.sum(axis=1, keepdims=True) + 2.0)
    pwm_window = pfm[args.start:args.end+1, :]
    plot_pwm(pwm_window, args.start, args.pwm_png, info_content=False)
    plot_pwm(pwm_window, args.start, args.pwm_info_png, info_content=True)

    
    bg_size = min(max(args.bg_size, max(128, args.samples)), len(Xall))
    bg_idx = rng.choice(len(Xall), size=bg_size, replace=False)
    bg = Xall[bg_idx].astype(np.float32)
    bg_mean = bg.mean(axis=0, keepdims=True)
    print(f"[DEBUG] Background size: {bg.shape[0]}")

    
    shap_values = None
    try:
        explainer = shap.DeepExplainer(logits_model, bg)
        shap_values = explainer.shap_values(X)
        print("[DEBUG] Used DeepExplainer.")
    except Exception as e:
        print(f"[WARN] DeepExplainer failed: {e}")
        try:
            explainer = shap.GradientExplainer(logits_model, bg)
            shap_values = explainer.shap_values(X)
            print("[DEBUG] Used GradientExplainer.")
        except Exception as e2:
            print(f"[WARN] GradientExplainer failed: {e2}")
            print("[INFO] Falling back to Integrated Gradients...")

    if shap_values is None:
        
        if args.ig_baseline == 'zero':
            baseline = np.zeros_like(X, dtype=np.float32)
        elif args.ig_baseline == 'uniform':
            baseline = np.full_like(X, 0.25, dtype=np.float32)
        else:  
            baseline = np.repeat(bg_mean, repeats=X.shape[0], axis=0).astype(np.float32)

        def integrated_gradients(inputs: np.ndarray, steps: int = 64) -> np.ndarray:
            inputs_tf = tf.convert_to_tensor(inputs, dtype=tf.float32)
            baseline_tf = tf.convert_to_tensor(baseline, dtype=tf.float32)
            total_grads = tf.zeros_like(inputs_tf)
            for k in range(1, steps + 1):
                alpha = tf.cast(k, tf.float32) / tf.cast(steps, tf.float32)
                x = baseline_tf + alpha * (inputs_tf - baseline_tf)
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    y = logits_model(x, training=False)
                    preds = select_class_output(y, args.class_index)
                grads = tape.gradient(preds, x)
                grads = tf.where(tf.math.is_finite(grads), grads, tf.zeros_like(grads))
                total_grads += grads
            avg_grads = total_grads / tf.cast(steps, tf.float32)
            ig = (inputs_tf - baseline_tf) * avg_grads
            return ig.numpy()

        class_attr = integrated_gradients(X)
    else:

        class_attr = shap_values[int(args.class_index)] if isinstance(shap_values, list) else shap_values

    print("[DEBUG] Attribution stats:",
          "min=", float(np.nanmin(class_attr)),
          "max=", float(np.nanmax(class_attr)),
          "mean=", float(np.nanmean(class_attr)),
          "abs-mean=", float(np.nanmean(np.abs(class_attr))))

    cwm = np.zeros((args.seq_len, 4), dtype=np.float64)
    counts = np.zeros((args.seq_len, 4), dtype=np.float64)
    for i in range(n_samples):
        rows = np.arange(args.seq_len)
        cols = base_idx[i]
        vals = class_attr[i, rows, cols]
        vals = np.where(np.isfinite(vals), vals, 0.0)
        cwm[rows, cols] += vals
        counts[rows, cols] += 1
    cwm = np.divide(cwm, counts, out=np.zeros_like(cwm), where=counts > 0)

    cwm_window = cwm[args.start:args.end+1, :]
    plot_signed_cwm(cwm_window, args.start, args.cwm_png)

    print(f"[DONE] Generated:\n  {args.cwm_png}\n  {args.pwm_png}\n  {args.pwm_info_png}")
    print("[NOTE] If CWM is still flat: confirm grad L2-norm > 0, try --ig_baseline uniform, and ensure --class_index 0 for binary models.")

if __name__ == "__main__":
    main()
