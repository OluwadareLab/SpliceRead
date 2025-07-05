# SpliceREAD

A deep learning framework for canonical and non-canonical splice site classification with synthetic data augmentation and SHAP-based interpretability.

[See SpliceRead Wiki for Full Documentation on Installation and Usage](https://github.com/OluwadareLab/SpliceRead/wiki)

**Authors:**

* Khushali Samderiya
* Rohit Menon
* Sahil Thapa
* Prof. Oluwatosin Oluwadare

___________________
#### OluwadareLab, University of Colorado, Colorado Springs
___________________


## Overview

**SpliceREAD** is a deep learning model for accurate splice site classification, especially focused on improving the detection of non-canonical splice sites. Traditional models often overlook these rare variants due to extreme class imbalance and sequence variability. SpliceREAD addresses this gap by combining residual convolutional neural networks with synthetic data augmentation using ADASYN, enabling improved learning of subtle genomic signals. The framework is robust across multiple species, sequence lengths, and validation settings, consistently outperforming existing tools in both canonical and non-canonical detection tasks.

---

## Folder Structure

```
SpliceRead/
+-- data/                 # Placeholder folder to be replaced with downloaded dataset
+-- models/               # Placeholder folder to be replaced with pretrained models
+-- output/               # Stores generated synthetic sequences and visualization outputs
+-- scripts/              # All training, generation, evaluation, and visualization scripts
¦   +-- training/         # Classifier training logic
¦   +-- data_augmentation/  # Synthetic data generation logic
¦   +-- evaluation/       # Performance evaluation scripts
¦   +-- visualization/    # Plotting utilities
¦   +-- interpretation/   # SHAP-based interpretation utilities
+-- Dockerfile            # Containerized environment for reproducibility
+-- README.md             # Project documentation
```

---

## Setup Instructions

### Step 1: Clone Repository

```bash
git clone https://github.com/OluwadareLab/SpliceRead.git
cd SpliceRead
```

### Step 2: Download Data and Models

Download the Zenodo archive from the link below:

[https://doi.org/10.5281/zenodo.15538290](https://doi.org/10.5281/zenodo.15538290)

### Step 3: Place Files

* Extract the `SpliceRead_Files.zip` archive.
* Replace the `data/` folder in the repo with the extracted `data/` folder.
* Replace the `models/` folder in the repo with the extracted `models/` folder.

### Step 4: Build Docker Image

```bash
docker build -t spliceread .
```

### Step 5: Start Docker Container

```bash
docker run -it --name spliceread-container \
  -v /path/to/local/SpliceRead:/workspace/SpliceRead \
  spliceread /bin/bash
```

Inside the container:

```bash
cd /workspace/SpliceRead
```

---

## Execution Instructions

### 1.Train & Evaluate

**Purpose**: Trains and evaluate the SpliceREAD model using 5-fold cross-validation.

**Arguments**:

| Flag                              | Description                                                      |
|-----------------------------------|------------------------------------------------------------------|
| `--three_class_no_synthetic`      | Use to train the model **without** synthetic data                |
| `--three_class`                   | Use to train the **with** synthetic data (default)               |
| `--synthetic_ratio <float>`       | % ratio of non-canonical to canonical sequences (default: 100.0) |
| `--use_smote`                     | Generate synthetic data with **SMOTE** (instead of ADASYN)       |
| `--show_progress`                 | Display progress bars during data loading                        |
| `--sequence_length <400\|600>`    | Sequence length in bp (default: 600)                             |
| `--train_dir <path>`              | Directory for training data                                      |
| `--test_dir <path>`               | Directory for test data                                          |
| `--model_dir <path>`              | Where trained models are saved                                   |
| `--output_dir <path>`             | Where evaluation outputs & logs go                               |
| `--model_path <path>`             | Path to a pre-trained model for evaluation                       |
| `--folds <int>`                   | Number of cross-validation folds (default: 5)                    |



**Command**:

```bash
python3 run.py [options]
```

**Example**:

```bash
python3 run.py --three-class \
  --synthetic_ratio 100 \
  --sequence_length 600 \
  --train_dir ./to_zenodo/data_600bp/train \
  --test_dir ./to_zenodo/data_600bp/test \
  --model_dir ./trained_model \
  --output_dir ./output \
  --show_progress
```
To evaluate a pretrained 600 bp model, download the model_files archive from Zenodo.

**Arguments**:
| Flag                          | Description                                                          |
|-------------------------------|----------------------------------------------------------------------|
| `--model_path <path>`         | Path to the trained model file (HDF5, e.g. `.h5`) (required)         |
| `--test_data <path>`          | Directory containing test sequences organized by class (required)    |
| `--out_dir <path>`            | Directory to save evaluation report (default: `./evaluation_results`)|
| `--sequence_length <600>`     | Sequence length in base pairs (default: `600`)                       |
| `--show_progress`             | Show progress bars while loading test data    

**Example**
```bash
python3 scripts/evaluation/run_evaluation.py \
  --model_path ./to_zenodo/model_files/best_model_fold_5.h5 \
  --test_data to_zenodo/data_600bp/test \
  --out_dir ./my_eval_results \
  --show_progress
```

---

### 2.`Visualization`
**Purpose**: Visualizes GC/AT nucleotide content using scatter plots.
To visualize the GC/AT nucleotied, first we have to generate the synthetic sequence using ADASYN.

#### i. `run_generator.py`

**Purpose**: Generates synthetic non-canonical sequences using ADASYN in one-hot encoding space.

**Input**:

* `data/train/POS/ACC/NC`
* `data/train/POS/DON/NC`

**Output**:

* `data/train/POS/ACC/ADASYN_TEST`
* `data/train/POS/DON/ADASYN_TEST`

**Command**:

```bash
python3 scripts/data_augmentation/run_generator.py
```

---

### 3. `run_feature_space_adasyn.py`

**Purpose**: Uses GC/AT content to generate synthetic sequences with ADASYN.

**Input**:

* Canonical and Non-canonical sequences from:

  * `data/train/POS/ACC/CAN` and `data/train/POS/ACC/NC`
  * `data/train/POS/DON/CAN` and `data/train/POS/DON/NC`

**Output**:

* Synthetic samples saved to:

  * `data/train/POS/ACC/ADASYN_SYN`
  * `data/train/POS/DON/ADASYN_SYN`

**Command**:

```bash
python3 scripts/data_augmentation/run_feature_space_adasyn.py
```

---

### 4. `run_visualization.py`

**Purpose**: Visualizes GC/AT nucleotide content using scatter plots.

**Input**:

* Canonical: `data/train/POS/ACC/CAN`
* Non-Canonical: `data/train/POS/ACC/NC`
* Synthetic: `data/train/POS/ACC/ADASYN_SYN`

**Output**:

* `Acceptor_Sequences_Canonical_vs_Non_Canonical.png`
* `Acceptor_Sequences_Canonical_vs_Combined_Non_Canonical.png`

**Command**:

```bash
python3 scripts/visualization/run_visualization.py \
  --canonical data/train/POS/ACC/CAN \
  --noncanonical data/train/POS/ACC/NC \
  --synthetic data/train/POS/ACC/ADASYN_SYN \
  --title "Acceptor Sequences"
```

---

### 5. `run_shap_logo.py`

**Purpose**: Computes SHAP values and generates sequence logo plots to interpret model decisions.

**Input**:

* Trained model: `models/SpliceRead_model.h5`
* Data folder: `data/train/POS/ACC`

**Output**:

* SHAP logo plot: `output/acceptor_shap_logo.png`

**Command**:

```bash
python3 scripts/interpretation/run_shap_logo.py \
  --model models/SpliceRead_model.h5 \
  --data data/train/POS/ACC \
  --samples 100 \
  --class_index 1 \
  --output output/acceptor_shap_logo.png
```

---

## Output Summary

* **Synthetic Sequences**  `data/train/POS/[ACC|DON]/ADASYN_SYN`
* **Trained Models**  `model_files/`
* **Visualizations**  `output/`

  * GC/AT Content Plots
  * SHAP Sequence Logos

---

## Citation

If you use SpliceREAD in your research, please cite our repository:

**Zenodo DOI**: [https://doi.org/10.5281/zenodo.15538290](https://doi.org/10.5281/zenodo.15538290)

---

## Contact

For questions, please contact [Khushali Samderiya](mailto:ksamderi@uccs.edu) or [Prof. Oluwatosin Oluwadare](mailto:ooluwada@uccs.edu)

---

## License

MIT License. See `LICENSE` file for details.
