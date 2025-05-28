
# ?? SpliceREAD

**SpliceREAD** is a deep learning-based framework designed to enhance splice site prediction, with a special focus on identifying **non-canonical splice sites**. The system integrates **residual CNN architectures** and **synthetic data augmentation using ADASYN**, accompanied by robust evaluation and interpretability pipelines using SHAP-based sequence logos.

---

## ?? Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Data Augmentation](#data-augmentation)
  - [Visualization](#visualization)
  - [SHAP Interpretation](#shap-interpretation)
- [Results](#results)
- [Authors](#authors)
- [License](#license)

---

## ?? Overview

SpliceREAD tackles the problem of **splice site classification** by using a deep CNN with **residual blocks**, specially tailored for recognizing both **canonical and non-canonical** sequences. In particular, it leverages **ADASYN** to synthesize additional examples of rare non-canonical splice sites, improving model generalization.

---

## ? Key Features

- ? CNN model with **residual connections** for sequence classification
- ? **Synthetic sequence generation** using ADASYN & SMOTE for imbalanced classes
- ? Detailed performance analysis (confusion matrix, accuracy, precision, recall)
- ? **SHAP-based interpretation** to visualize sequence importance
- ? GC/AT Content **nucleotide visualizations** using scatter plots
- ? Modular design and command-line support for reproducibility

---

## ?? Directory Structure

```
SpliceREAD/
+-- model/                # CNN model definitions and training
+-- data_augmentation/   # Synthetic sequence generation (ADASYN & SMOTE)
+-- visualization/       # GC/AT content scatter plot generation
+-- interpretation/      # SHAP-based sequence logo visualizations
+-- data/                # Canonical / Non-Canonical datasets
+-- scripts/             # Combined training and evaluation scripts
+-- README.md            # You're here
```

---

## ?? Installation

```bash
git clone https://github.com/your_username/SpliceREAD.git
cd SpliceREAD

# Recommended: use virtualenv or conda
pip install -r requirements.txt
```

> ?? This project requires **TensorFlow >= 2.x**, **imblearn**, **matplotlib**, **logomaker**, and **SHAP**.

---

## ?? Usage

### ?? Training

```bash
python3 run.py \
  --data /path/to/your/training/data \
  --output model_output/SpliceREAD_model.h5
```

### ?? Evaluation

```bash
python3 evaluate.py \
  --model model_output/SpliceREAD_model.h5 \
  --test_data /path/to/test/data
```

### ?? Data Augmentation

```bash
python3 run_generator.py \
  --noncanonical_acceptor_folder /.../ACC/NC \
  --synthetic_acceptor_folder /.../ACC/ADASYN_SYN \
  --noncanonical_donor_folder /.../DON/NC \
  --synthetic_donor_folder /.../DON/ADASYN_SYN \
  --n_synthetic_acceptor 8714 \
  --n_synthetic_donor 8399
```

### ?? Visualization

```bash
python3 run_visualization.py \
  --canonical /path/to/CAN \
  --noncanonical /path/to/NC \
  --synthetic /path/to/ADASYN_SYN \
  --title "Acceptor Sequences"
```

### ?? SHAP Interpretation

```bash
python3 run_shap_logo.py \
  --model model_output/SpliceREAD_model.h5 \
  --data /path/to/DATASET/TRAIN/POS/ACC \
  --samples 100 \
  --class_index 1 \
  --output acceptor_shap_logo.png
```

---

## ?? Results

Our experiments show that **SpliceREAD**:
- Improves **non-canonical splice site classification** by over 15% using synthetic data
- Preserves high **precision and recall** even under imbalanced datasets
- Provides **interpretable insights** into sequence motifs using SHAP

Refer to our full paper for benchmarking and ablation studies.

---

## ????? Authors

**SpliceREAD** was developed by:

- **Khushali Samderiya**  
- **Rohit Menon**  
- **Sahil Thapa**  
- **Prof. Oluwatosin Oluwadare**  

*Affiliation*: [Bioinformatics Lab, University of Colorado Colorado Springs (UCCS)](https://bioinformatics.uccs.edu)

---

## ?? License

This project is released under the MIT License.
