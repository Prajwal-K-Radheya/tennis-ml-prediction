# ATP Tennis ML Project — Data Cleaning & Train/Test Split

This repository contains the complete preprocessing, historical feature engineering, and chronological dataset split for the **ATP Tennis Match Prediction** Machine Learning project.

---

## 📌 Target Handoff Summary

The dataset has been cleaned and engineered with pre-match historical performance metrics (with zero future-data leakage) and split chronologically:

| Dataset | Period | Rows | Purpose |
| :--- | :--- | :--- | :--- |
| **`data/train_data.csv`** | 2018–2023 | **31,084** | Model Training & Hyperparameter Tuning |
| **`data/test_data.csv`** | 2024 | **6,060** | Final Out-of-Time Model Evaluation |
| **`data/cleaned_atp_data.csv`** | 2018–2024 | **18,572** | Cleaned Raw Match Records |

---

## 🛠️ Project Structure

```text
├── data/
│   ├── train_data.csv                              # Chronological training split (2018-2023)
│   ├── test_data.csv                               # Chronological test split (2024)
│   └── cleaned_atp_data.csv                        # Deduplicated & cleaned match data
├── notebooks/
│   └── data_cleaning_and_feature_engineering.ipynb # Fully executed Jupyter notebook with outputs
├── src/
│   └── data_preprocessing.py                       # Standalone pipeline execution script
├── atp_matches_2018.csv ... atp_matches_2024.csv   # Raw ATP match datasets (7 years)
├── ATP_Data_Cleaning_to_Dataset_Split_Guide.pdf    # Project specification & teammate guide
├── requirements.txt                                # Python dependencies
└── README.md                                       # Project documentation
```

---

## ⚙️ Feature Engineering & Leakage Prevention Policy

### 1. Chronological Rolling Features
Features are computed using **only information available before each match**. For every match:
- Player historical win rate (`win_rate_diff`)
- Surface-specific historical win rate (`surface_rate_diff`)
- Head-to-head records (`h2h_diff`)

Historical statistics are updated only **after** the match features have been logged.

### 2. Dual Perspective (Balanced Target)
Each match is recorded twice to prevent winner bias:
- **Perspective 1 (Winner)**: `player1` = Winner, `player2` = Loser, `target` = 1
- **Perspective 2 (Loser)**: `player1` = Loser, `player2` = Winner, `target` = 0

### 3. Feature Set
- `rank_diff`: Difference in ATP ranking (`rank1 - rank2`)
- `rank_points_diff`: Difference in ranking points (`points1 - points2`)
- `age_diff`: Difference in player age (`age1 - age2`)
- `win_rate_diff`: Difference in cumulative match win rate
- `surface_rate_diff`: Difference in cumulative surface win rate (Hard, Clay, Grass, Carpet)
- `h2h_diff`: Difference in cumulative head-to-head wins
- `surface`: Categorical match surface (`Hard`, `Clay`, `Grass`, etc.)
- `year`: Match calendar year (used for chronological splitting)
- `target`: Binary outcome (1 = `player1` won, 0 = `player2` won)

> **Note on Data Leakage:** Post-match statistics (aces, double faults, break points won, first-serve percentage) are strictly excluded from model input features because they cannot be observed prior to match start.

---

## 🚀 How to Run

### Installation
```bash
pip install -r requirements.txt
```

### Run Data Preprocessing Pipeline
```bash
python src/data_preprocessing.py
```

### Run Jupyter Notebook
```bash
jupyter notebook notebooks/data_cleaning_and_feature_engineering.ipynb
```

---

## 🎯 Model Training Starting Point

For the model-training teammate:
1. Load `data/train_data.csv` and `data/test_data.csv`.
2. Separate features $X$ and target $y$ (`target`).
3. Preprocess categorical features (e.g., One-Hot Encoding for `surface`).
4. Train baseline and benchmark models:
   - Logistic Regression
   - Random Forest Classifier
   - Support Vector Machine (SVM)
