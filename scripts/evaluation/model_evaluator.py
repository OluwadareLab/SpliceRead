import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.multiclass import unique_labels
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D, BatchNormalization

class ResidualBlock(layers.Layer):
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
            x = layers.Activation('relu')(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return layers.Add()([inputs, x])

    def get_config(self):
        return {
            **super().get_config(),
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "use_activation": self.use_activation
        }

NUCLEOTIDE_MAP = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}

def one_hot_encode(sequence):
    return np.array([NUCLEOTIDE_MAP[nuc] for nuc in sequence])

def load_sequences_from_folder(folder_path, label):
    data, labels = [], []
    if os.path.exists(folder_path):
        for fname in os.listdir(folder_path):
            with open(os.path.join(folder_path, fname), 'r') as f:
                seq = f.read().strip()
                if len(seq) == 600:
                    data.append(one_hot_encode(seq))
                    labels.append(label)
    else:
        print(f"[WARN] Missing folder: {folder_path}")
    return np.array(data), np.array(labels)

def load_test_data(base_path):
    acc_path = os.path.join(base_path, 'POS', 'ACC')
    don_path = os.path.join(base_path, 'POS', 'DON')
    acc_can, acc_can_y = load_sequences_from_folder(os.path.join(acc_path, 'CAN'), 0)
    acc_nc, acc_nc_y = load_sequences_from_folder(os.path.join(acc_path, 'NC'), 1)
    don_can, don_can_y = load_sequences_from_folder(os.path.join(don_path, 'CAN'), 2)
    don_nc, don_nc_y = load_sequences_from_folder(os.path.join(don_path, 'NC'), 3)
    X = np.concatenate([acc_can, acc_nc, don_can, don_nc])
    y = np.concatenate([acc_can_y, acc_nc_y, don_can_y, don_nc_y])
    return X, y

def evaluate_model(model_path, X, y):
    model = load_model(model_path, custom_objects={"ResidualBlock": ResidualBlock})
    preds = model.predict(X)
    preds_cls = np.argmax(preds, axis=1)

    accuracy = accuracy_score(y, preds_cls)
    f1 = f1_score(y, preds_cls, average='weighted')
    precision = precision_score(y, preds_cls, average='weighted')
    recall = recall_score(y, preds_cls, average='weighted')

    unique_classes = sorted(unique_labels(y, preds_cls))
    default_names = ["Acceptor CAN", "Acceptor NC", "Donor CAN", "Donor NC"]
    target_names = [default_names[i] if i < len(default_names) else f"Class {i}" for i in unique_classes]

    report = classification_report(y, preds_cls, labels=unique_classes, target_names=target_names)

    return accuracy, f1, precision, recall, report
