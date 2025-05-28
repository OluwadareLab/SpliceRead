import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from scripts.models.cnn_classifier import deep_cnn_classifier

def k_fold_cross_validation(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracy_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        print(f"\nTraining on fold {fold}/{k}...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = to_categorical(y[train_idx], 3), to_categorical(y[val_idx], 3)
        model = deep_cnn_classifier(X_train.shape[1])
        checkpoint = ModelCheckpoint(f"best_model_fold_{fold}.h5", monitor='val_accuracy',
                                     save_best_only=True, mode='max', verbose=1)
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=20, batch_size=32, callbacks=[checkpoint], verbose=1)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        accuracy_scores.append(val_acc)
        print(f"Fold {fold} Accuracy: {val_acc:.4f}")
    print(f"\nAverage Accuracy across {k} folds: {np.mean(accuracy_scores):.4f}")
