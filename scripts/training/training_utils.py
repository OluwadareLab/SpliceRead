import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import to_categorical
from scripts.models.cnn_classifier import deep_cnn_classifier
from scripts.data_augmentation.generator import sequence_to_onehot, onehot_to_sequence, apply_adasyn, apply_smote
import tensorflow as tf
import os
import math

# Configure GPU memory growth to prevent OOM errors
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] Configured {len(gpus)} GPUs with memory growth enabled")
        
        # Use MirroredStrategy only if multiple GPUs are available
        if len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            print(f"[INFO] Using MirroredStrategy for {len(gpus)} GPUs")
        else:
            strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
            print(f"[INFO] Using OneDeviceStrategy for single GPU")
    else:
        print("[INFO] No GPUs found, using CPU")
        strategy = tf.distribute.OneDeviceStrategy("/CPU:0")
        
    print(f"Number of devices: {strategy.num_replicas_in_sync}")
    
except RuntimeError as e:
    print(f"[WARNING] GPU configuration failed: {e}")
    # Fallback to CPU strategy
    strategy = tf.distribute.OneDeviceStrategy("/CPU:0")
    print("[INFO] Falling back to CPU strategy")


def one_hot_encode(sequence):
    """Convert DNA sequence to one-hot encoding."""
    base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((len(sequence), 4))
    for i, base in enumerate(sequence):
        if base in base_to_index:
            one_hot[i, base_to_index[base]] = 1
    return one_hot

def one_hot_to_sequence(one_hot_array):
    """Convert one-hot encoding back to DNA sequence."""
    index_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    sequence = ""
    for position in one_hot_array:
        base_index = np.argmax(position)
        sequence += index_to_base[base_index]
    return sequence

def generate_synthetic_for_fold(X_train, y_train, ratio=10.0, use_smote=False):
    
    print(f"[INFO] Generating synthetic data for this fold (ratio: {ratio}%)")
    
    # Separate by class (0=Acceptor, 1=Donor, 2=Negative)
    acceptor_mask = (y_train == 0)
    donor_mask = (y_train == 1)
    
    X_acc = X_train[acceptor_mask]
    X_don = X_train[donor_mask]
    
    if len(X_acc) == 0 or len(X_don) == 0:
        print("[WARN] No acceptor or donor sequences found in this fold")
        return np.array([]).reshape(0, X_train.shape[1], X_train.shape[2]), np.array([])
    
    
    synthetic_sequences = []
    synthetic_labels = []
    
    for class_label, X_class, class_name in [(0, X_acc, "Acceptor"), (1, X_don, "Donor")]:
        if len(X_class) < 2:
            print(f"[WARN] Not enough {class_name} sequences ({len(X_class)}) to generate synthetic data")
            continue
            
        # Calculate target count
        current_count = len(X_class)
        target_count = math.ceil((ratio / 100) * current_count)
        needed = max(0, target_count - current_count)
        
        if needed == 0:
            print(f"[INFO] {class_name}: No synthetic sequences needed ({current_count} >= {target_count})")
            continue
            
        print(f"[INFO] {class_name}: Generating {needed} synthetic sequences (current: {current_count}, target: {target_count})")
        
        try:
            X_flat = X_class.reshape(len(X_class), -1)
            
            # Generate synthetic data
            if use_smote:
                X_synthetic_flat = apply_smote(X_flat, current_count + needed)
            else:
                X_synthetic_flat = apply_adasyn(X_flat, current_count + needed)
            
            X_synthetic_flat = X_synthetic_flat[-needed:]
            
            # Reshape back to original format
            X_synthetic_class = X_synthetic_flat.reshape(needed, X_train.shape[1], X_train.shape[2])
            y_synthetic_class = np.full(needed, class_label)
            
            synthetic_sequences.append(X_synthetic_class)
            synthetic_labels.append(y_synthetic_class)
            
            print(f"[INFO] {class_name}: Successfully generated {len(X_synthetic_class)} synthetic non-canonical sequences")
            
        except Exception as e:
            if not use_smote:
                print(f"[WARN] ADASYN failed for {class_name}: {e}. Retrying with SMOTE...")
                try:
                    X_flat = X_class.reshape(len(X_class), -1)
                    X_synthetic_flat = apply_smote(X_flat, current_count + needed)
                    X_synthetic_flat = X_synthetic_flat[-needed:]
                    X_synthetic_class = X_synthetic_flat.reshape(needed, X_train.shape[1], X_train.shape[2])
                    y_synthetic_class = np.full(needed, class_label)
                    synthetic_sequences.append(X_synthetic_class)
                    synthetic_labels.append(y_synthetic_class)
                    print(f"[INFO] {class_name}: Successfully generated {len(X_synthetic_class)} synthetic non-canonical sequences with SMOTE")
                except Exception as se:
                    print(f"[ERROR] SMOTE also failed for {class_name}: {se}")
            else:
                print(f"[ERROR] SMOTE failed for {class_name}: {e}")
    
    if len(synthetic_sequences) == 0:
        print("[WARN] No synthetic data generated")
        return np.array([]).reshape(0, X_train.shape[1], X_train.shape[2]), np.array([])
    
    # Combine all synthetic data
    X_synthetic = np.concatenate(synthetic_sequences)
    y_synthetic = np.concatenate(synthetic_labels)
    
    print(f"[INFO] Total synthetic non-canonical data generated: {len(X_synthetic)} sequences")
    return X_synthetic, y_synthetic


def analyze_fold_composition(X_train, y_train, X_synthetic=None, y_synthetic=None, fold_num=1):
    print(f"\n========== FOLD {fold_num} TRAINING SET COMPOSITION ==========")
    
    # Basic class counts
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    total_sequences = len(y_train)
    
    print(f"Total Training Sequences: {total_sequences}")
    
    for cls, count in zip(unique_classes, class_counts):
        if cls == 0:
            class_name = "Acceptor"
        elif cls == 1:
            class_name = "Donor"
        elif cls == 2:
            class_name = "Negative"
        else:
            class_name = f"Class {cls}"
        
        percentage = (count / total_sequences) * 100
        print(f"  {class_name}: {count} sequences ({percentage:.1f}%)")
    
    if X_synthetic is not None and len(X_synthetic) > 0:
        print(f"\nSynthetic Data Added: {len(X_synthetic)} sequences")
        base_sequences = total_sequences - len(X_synthetic)
        print(f"Base Data: {base_sequences} sequences")
        
        # Analyze synthetic by class
        unique_syn_classes, syn_class_counts = np.unique(y_synthetic, return_counts=True)
        for cls, count in zip(unique_syn_classes, syn_class_counts):
            if cls == 0:
                print(f"  Synthetic Acceptor: {count} sequences")
            elif cls == 1:
                print(f"  Synthetic Donor: {count} sequences")
    
    print("=" * 60)

def k_fold_cross_validation(X_base, y_base, k=5, model_dir='./model_output', 
                          use_synthetic=False, synthetic_ratio=10.0, use_smote=False,
                          # Legacy parameters for backward compatibility
                          X_synthetic=None, y_synthetic=None, output_dir='./results'):
   
    num_classes = 3  
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracy_scores = []
    val_loss_scores = []
    training_loss_scores = []
    training_acc_scores = []
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] Starting {k}-fold cross-validation...")
    print(f"[INFO] Base data shape: {X_base.shape}, Labels shape: {y_base.shape}")
    
    if use_synthetic and X_synthetic is None:
        print(f"[INFO] Will generate synthetic data per fold (ratio: {synthetic_ratio}%)")
        print(f"[INFO] Generation method: {'SMOTE' if use_smote else 'ADASYN'}")
    elif X_synthetic is not None and len(X_synthetic) > 0:
        print(f"[INFO] Using pre-loaded synthetic data (legacy mode)")
        print(f"[INFO] Pre-loaded synthetic breakdown:")
        unique_syn_classes, syn_counts = np.unique(y_synthetic, return_counts=True)
        for cls, count in zip(unique_syn_classes, syn_counts):
            if cls == 0:
                print(f"  Pre-loaded Synthetic Acceptor: {count} sequences")
            elif cls == 1:
                print(f"  Pre-loaded Synthetic Donor: {count} sequences")
    else:
        print(f"[INFO] No synthetic data will be used")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_base), start=1):
        print(f"\nTraining on fold {fold}/{k}...")
        
        X_train_base, X_val = X_base[train_idx], X_base[val_idx]
        y_train_base, y_val = y_base[train_idx], y_base[val_idx]
        
        
        X_synthetic_fold, y_synthetic_fold = None, None
        
        if use_synthetic:
            # Generate synthetic data for this specific fold
            X_synthetic_fold, y_synthetic_fold = generate_synthetic_for_fold(
                X_train_base, y_train_base, 
                ratio=synthetic_ratio, 
                use_smote=use_smote
            )
            
            if len(X_synthetic_fold) > 0:
                X_train = np.concatenate([X_train_base, X_synthetic_fold])
                y_train = np.concatenate([y_train_base, y_synthetic_fold])
                method = f"Per-fold {'SMOTE' if use_smote else 'ADASYN'}"
            else:
                X_train = X_train_base
                y_train = y_train_base
                X_synthetic_fold = None
                y_synthetic_fold = None
                method = "None (generation failed)"
                
        elif X_synthetic is not None and len(X_synthetic) > 0:
            X_train = np.concatenate([X_train_base, X_synthetic])
            y_train = np.concatenate([y_train_base, y_synthetic])
            X_synthetic_fold, y_synthetic_fold = X_synthetic, y_synthetic
            method = "Pre-loaded ADASYN"
        else:
            X_train = X_train_base
            y_train = y_train_base
            X_synthetic_fold = None
            y_synthetic_fold = None
            method = "None"
        
        show_detailed_fold_stats(
            fold, X_train_base, y_train_base, X_val, y_val,
            X_synthetic_fold, y_synthetic_fold, method
        )
        
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)
        
        # Train model
        with strategy.scope():
            model = deep_cnn_classifier(X_train.shape[1], num_classes)
        
        checkpoint = ModelCheckpoint(
            os.path.join(model_dir, f"best_model_fold_{fold}.h5"), 
            monitor='val_accuracy',
            save_best_only=True, 
            mode='max', 
            verbose=1
        )
        
        csv_logger = CSVLogger(
            os.path.join(output_dir, f"training_log_fold_{fold}.csv"),
            append=False  # Create new file for each fold
        )
        
        history = model.fit(X_train, y_train_cat, 
                  validation_data=(X_val, y_val_cat),
                  epochs=40, 
                  batch_size=32, 
                  callbacks=[checkpoint, csv_logger], 
                  verbose=1)
        
       
        final_train_loss = history.history['loss'][-1]
        final_train_acc = history.history['accuracy'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        training_loss_scores.append(final_train_loss)
        training_acc_scores.append(final_train_acc)
        val_loss_scores.append(final_val_loss)
        accuracy_scores.append(final_val_acc)
        
        print(f"Fold {fold} - Final Training: Loss={final_train_loss:.4f}, Acc={final_train_acc:.4f}")
        print(f"Fold {fold} - Final Validation: Loss={final_val_loss:.4f}, Acc={final_val_acc:.4f}")
    
    save_cross_fold_summary(
        training_loss_scores, training_acc_scores, val_loss_scores, accuracy_scores,
        k, output_dir, use_synthetic, synthetic_ratio, use_smote
    )
    
    print(f"\nCross-Fold Summary ({k} folds):")
    print(f"  Validation Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    print(f"  Validation Loss: {np.mean(val_loss_scores):.4f} ± {np.std(val_loss_scores):.4f}")
    print(f"  Training Accuracy: {np.mean(training_acc_scores):.4f} ± {np.std(training_acc_scores):.4f}")
    print(f"  Training Loss: {np.mean(training_loss_scores):.4f} ± {np.std(training_loss_scores):.4f}")
    
    if use_synthetic:
        print(f"[INFO] Synthetic data was generated per fold (ratio: {synthetic_ratio}%)")
    elif X_synthetic is not None and len(X_synthetic) > 0:
        print(f"[INFO] Pre-loaded synthetic data was used in training folds only")
    else:
        print(f"[INFO] No synthetic data was used")

def k_fold_cross_validation_old_signature(*args, **kwargs):
    pass  # Function removed as legacy

def generate_synthetic_for_fold_correct(*args, **kwargs):
    pass  # Function removed as legacy

def generate_synthetic_for_fold_from_separated_data(*args, **kwargs):
    pass  # Function removed as legacy

def estimate_canonical_noncanonical_breakdown(*args, **kwargs):
    pass  # Function removed as legacy

def show_detailed_fold_stats(*args, **kwargs):
    pass  # Function removed as legacy

def k_fold_cross_validation_with_separated_data(
    acc_can_data, acc_nc_data, don_can_data, don_nc_data, neg_data,
    k=5, model_dir='./model_output', use_synthetic=False, synthetic_ratio=10.0, use_smote=False, output_dir='./results'):
    all_sequences = []
    all_labels = []
    sequence_types = []  
    
    all_sequences.extend(acc_can_data)
    all_labels.extend([0] * len(acc_can_data))
    sequence_types.extend(['acc_can'] * len(acc_can_data))
    
    all_sequences.extend(acc_nc_data)
    all_labels.extend([0] * len(acc_nc_data))
    sequence_types.extend(['acc_nc'] * len(acc_nc_data))
    
    all_sequences.extend(don_can_data) 
    all_labels.extend([1] * len(don_can_data))
    sequence_types.extend(['don_can'] * len(don_can_data))
    
    all_sequences.extend(don_nc_data)
    all_labels.extend([1] * len(don_nc_data))
    sequence_types.extend(['don_nc'] * len(don_nc_data))
    
    all_sequences.extend(neg_data)
    all_labels.extend([2] * len(neg_data))
    sequence_types.extend(['neg'] * len(neg_data))
    
    X_all = np.array(all_sequences)
    y_all = np.array(all_labels)
    types_all = np.array(sequence_types)
    
    print(f"[INFO] Total data for CV: {len(X_all)} sequences")
    print(f"  ACC/CAN: {len(acc_can_data)}")
    print(f"  ACC/NC: {len(acc_nc_data)}")
    print(f"  DON/CAN: {len(don_can_data)}")
    print(f"  DON/NC: {len(don_nc_data)}")
    print(f"  NEG: {len(neg_data)}")
    
    num_classes = 3  # Always 3 for this system
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracy_scores = []
    val_loss_scores = []
    training_loss_scores = []
    training_acc_scores = []
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all), start=1):
        print(f"\nTraining on fold {fold}/{k}...")
        
        # Split data
        X_train_base = X_all[train_idx]
        y_train_base = y_all[train_idx] 
        types_train = types_all[train_idx]
        
        X_val = X_all[val_idx]
        y_val = y_all[val_idx]
        
        # Separate training data by type
        acc_can_mask = (types_train == 'acc_can')
        acc_nc_mask = (types_train == 'acc_nc')
        don_can_mask = (types_train == 'don_can')
        don_nc_mask = (types_train == 'don_nc')
        neg_mask = (types_train == 'neg')
        
        X_acc_can_fold = X_train_base[acc_can_mask]
        X_acc_nc_fold = X_train_base[acc_nc_mask]
        X_don_can_fold = X_train_base[don_can_mask]
        X_don_nc_fold = X_train_base[don_nc_mask]
        X_neg_fold = X_train_base[neg_mask]
        
        # Show detailed fold composition
        print(f"\nDetailed Fold {fold} Composition:")
        print(f"  Training Acceptor Canonical: {len(X_acc_can_fold)}")
        print(f"  Training Acceptor Non-canonical: {len(X_acc_nc_fold)}")
        print(f"  Training Donor Canonical: {len(X_don_can_fold)}")
        print(f"  Training Donor Non-canonical: {len(X_don_nc_fold)}")
        print(f"  Training Negative: {len(X_neg_fold)}")
        print(f"  Validation Total: {len(X_val)}")
        
        # Generate synthetic data following run_generator.py logic
        X_synthetic_fold = []
        y_synthetic_fold = []
        
        if use_synthetic:
            print(f"\nGenerating synthetic data (ratio: {synthetic_ratio}%):")
            
            # Acceptor synthetic generation
            acc_can_count = len(X_acc_can_fold)
            acc_nc_count = len(X_acc_nc_fold)
            acc_target = math.ceil((synthetic_ratio / 100) * acc_can_count)
            acc_needed = max(0, acc_target - acc_nc_count)
            
            print(f"  Acceptor: Canonical={acc_can_count}, NC={acc_nc_count}, Target={acc_target}, Need={acc_needed}")
            
            if acc_needed > 0 and len(X_acc_nc_fold) >= 2:
                try:
                    X_flat = X_acc_nc_fold.reshape(len(X_acc_nc_fold), -1)
                    if use_smote:
                        X_syn_flat = apply_smote(X_flat, acc_nc_count + acc_needed)
                    else:
                        X_syn_flat = apply_adasyn(X_flat, acc_nc_count + acc_needed)
                    
                    X_syn_flat = X_syn_flat[-acc_needed:]
                    X_syn_acc = X_syn_flat.reshape(acc_needed, X_acc_nc_fold.shape[1], X_acc_nc_fold.shape[2])
                    
                    X_synthetic_fold.extend(X_syn_acc)
                    y_synthetic_fold.extend([0] * acc_needed)
                    print(f"  Generated {acc_needed} synthetic non-canonical acceptor sequences")
                except Exception as e:
                    print(f"  Failed to generate acceptor synthetic: {e}")
            
            # Donor synthetic generation
            don_can_count = len(X_don_can_fold)
            don_nc_count = len(X_don_nc_fold)
            don_target = math.ceil((synthetic_ratio / 100) * don_can_count)
            don_needed = max(0, don_target - don_nc_count)
            
            print(f"  Donor: Canonical={don_can_count}, NC={don_nc_count}, Target={don_target}, Need={don_needed}")
            
            if don_needed > 0 and len(X_don_nc_fold) >= 2:
                try:
                    X_flat = X_don_nc_fold.reshape(len(X_don_nc_fold), -1)
                    if use_smote:
                        X_syn_flat = apply_smote(X_flat, don_nc_count + don_needed)
                    else:
                        X_syn_flat = apply_adasyn(X_flat, don_nc_count + don_needed)
                    
                    X_syn_flat = X_syn_flat[-don_needed:]
                    X_syn_don = X_syn_flat.reshape(don_needed, X_don_nc_fold.shape[1], X_don_nc_fold.shape[2])
                    
                    X_synthetic_fold.extend(X_syn_don)
                    y_synthetic_fold.extend([1] * don_needed)
                    print(f"  Generated {don_needed} synthetic non-canonical donor sequences")
                except Exception as e:
                    print(f"  Failed to generate donor synthetic: {e}")
        
        # Combine training data
        if len(X_synthetic_fold) > 0:
            X_train = np.concatenate([X_train_base, np.array(X_synthetic_fold)])
            y_train = np.concatenate([y_train_base, np.array(y_synthetic_fold)])
            print(f"\nFinal training set: {len(X_train_base)} base + {len(X_synthetic_fold)} synthetic = {len(X_train)} total")
        else:
            X_train = X_train_base
            y_train = y_train_base
            print(f"\nFinal training set: {len(X_train)} base sequences only")
        
        # Convert to categorical and train
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)
        
        with strategy.scope():
            model = deep_cnn_classifier(X_train.shape[1], num_classes)
        
        checkpoint = ModelCheckpoint(
            os.path.join(model_dir, f"best_model_fold_{fold}.h5"), 
            monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
        )
        
        # CSV logger to record training metrics
        csv_logger = CSVLogger(
            os.path.join(output_dir, f"training_log_fold_{fold}.csv"),
            append=False  # Create new file for each fold
        )
        
        history = model.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat),
                  epochs=40, batch_size=32, callbacks=[checkpoint, csv_logger], verbose=1)
        
        # Collect final epoch metrics
        final_train_loss = history.history['loss'][-1]
        final_train_acc = history.history['accuracy'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        # Store metrics for summary
        training_loss_scores.append(final_train_loss)
        training_acc_scores.append(final_train_acc)
        val_loss_scores.append(final_val_loss)
        accuracy_scores.append(final_val_acc)
        
        print(f"Fold {fold} - Final Training: Loss={final_train_loss:.4f}, Acc={final_train_acc:.4f}")
        print(f"Fold {fold} - Final Validation: Loss={final_val_loss:.4f}, Acc={final_val_acc:.4f}")
    
    # Calculate and save cross-fold summary statistics
    save_cross_fold_summary(
        training_loss_scores, training_acc_scores, val_loss_scores, accuracy_scores,
        k, output_dir, use_synthetic, synthetic_ratio, use_smote
    )
    
    print(f"\nCross-Fold Summary ({k} folds):")
    print(f"  Validation Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    print(f"  Validation Loss: {np.mean(val_loss_scores):.4f} ± {np.std(val_loss_scores):.4f}")
    print(f"  Training Accuracy: {np.mean(training_acc_scores):.4f} ± {np.std(training_acc_scores):.4f}")
    print(f"  Training Loss: {np.mean(training_loss_scores):.4f} ± {np.std(training_loss_scores):.4f}")
    print(f"[INFO] Per-fold synthetic generation completed following run_generator.py logic")

def save_cross_fold_summary(train_loss_scores, train_acc_scores, val_loss_scores, val_acc_scores,
                           k_folds, output_dir, use_synthetic=False, synthetic_ratio=0.0, use_smote=False):
    
    import pandas as pd
    import datetime
    
    # Create summary data
    metrics_data = {
        'metric': ['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'],
        'mean': [
            np.mean(train_loss_scores),
            np.mean(train_acc_scores), 
            np.mean(val_loss_scores),
            np.mean(val_acc_scores)
        ],
        'std': [
            np.std(train_loss_scores),
            np.std(train_acc_scores),
            np.std(val_loss_scores), 
            np.std(val_acc_scores)
        ],
        'min': [
            np.min(train_loss_scores),
            np.min(train_acc_scores),
            np.min(val_loss_scores),
            np.min(val_acc_scores)
        ],
        'max': [
            np.max(train_loss_scores),
            np.max(train_acc_scores),
            np.max(val_loss_scores),
            np.max(val_acc_scores)
        ]
    }
    
    # Add individual fold values
    all_scores = [train_loss_scores, train_acc_scores, val_loss_scores, val_acc_scores]
    for fold in range(k_folds):
        fold_col = f'fold_{fold+1}'
        metrics_data[fold_col] = [scores[fold] for scores in all_scores]
    
    df = pd.DataFrame(metrics_data)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if use_synthetic:
        config_str = f"synthetic_{synthetic_ratio}pct_{'smote' if use_smote else 'adasyn'}"
    else:
        config_str = "no_synthetic"
    
    summary_file = os.path.join(output_dir, f"cross_fold_summary_{config_str}_{timestamp}.csv")
    df.to_csv(summary_file, index=False, float_format='%.6f')
    
    print(f"[INFO] Cross-fold summary saved to: {summary_file}")
    
    # Also save a simple summary with metadata
    metadata_file = os.path.join(output_dir, f"experiment_summary_{config_str}_{timestamp}.txt")
    with open(metadata_file, 'w') as f:
        f.write("SpliceRead Cross-Fold Validation Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of folds: {k_folds}\n")
        f.write(f"Synthetic data: {'Yes' if use_synthetic else 'No'}\n")
        if use_synthetic:
            f.write(f"Synthetic ratio: {synthetic_ratio}%\n")
            f.write(f"Synthetic method: {'SMOTE' if use_smote else 'ADASYN'}\n")
        f.write("\n")
        
        f.write("Cross-Fold Performance Summary:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Validation Accuracy: {np.mean(val_acc_scores):.4f} ± {np.std(val_acc_scores):.4f}\n")
        f.write(f"Validation Loss:     {np.mean(val_loss_scores):.4f} ± {np.std(val_loss_scores):.4f}\n")
        f.write(f"Training Accuracy:   {np.mean(train_acc_scores):.4f} ± {np.std(train_acc_scores):.4f}\n")
        f.write(f"Training Loss:       {np.mean(train_loss_scores):.4f} ± {np.std(train_loss_scores):.4f}\n")
        f.write("\n")
        
        f.write("Individual Fold Results:\n")
        f.write("-" * 25 + "\n")
        for fold in range(k_folds):
            f.write(f"Fold {fold+1}: Val_Acc={val_acc_scores[fold]:.4f}, Val_Loss={val_loss_scores[fold]:.4f}, "
                   f"Train_Acc={train_acc_scores[fold]:.4f}, Train_Loss={train_loss_scores[fold]:.4f}\n")
    
    print(f"[INFO] Experiment summary saved to: {metadata_file}")

    
