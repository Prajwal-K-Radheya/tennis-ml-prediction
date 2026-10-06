# ATP Tennis Match Prediction — Data Preparation

This repository contains the data cleaning, chronological historical feature engineering, and train/test split for ATP tennis match outcome prediction.

## Repository Structure

```text
├── data/
│   ├── raw/                      # Raw ATP match CSVs (2018–2024)
│   ├── train_data.csv            # Training split (2018–2023)
│   ├── test_data.csv             # Test split (2024)
│   └── cleaned_atp_data.csv      # Cleaned combined matches
├── docs/
│   └── ATP_Data_Cleaning_to_Dataset_Split_Guide.pdf
├── notebooks/
│   └── data_cleaning_and_feature_engineering.ipynb
├── src/
│   └── data_preprocessing.py
├── requirements.txt
└── README.md
```

## Dataset Summary

- **Training set (`train_data.csv`)**: 31,084 samples (matches from 2018 to 2023)
- **Testing set (`test_data.csv`)**: 6,060 samples (matches from 2024)
- **Target**: Binary classification (`1` = Player 1 wins, `0` = Player 2 wins), perfectly balanced 50/50.

## Feature Set

All features are calculated prior to match commencement to prevent future data leakage:
- `rank_diff`: Difference in ATP rank (`rank1 - rank2`)
- `rank_points_diff`: Difference in ranking points
- `age_diff`: Difference in player ages
- `win_rate_diff`: Rolling overall win rate differential
- `surface_rate_diff`: Rolling surface-specific win rate differential
- `h2h_diff`: Historical head-to-head win differential
- `surface`: Match surface (`Hard`, `Clay`, `Grass`)
- `year`: Match year
- `target`: Match winner indicator

Post-match statistics (aces, break points, service stats) are intentionally excluded as they are not known before a match starts.

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing Pipeline
```bash
python src/data_preprocessing.py
```

### 3. Open Jupyter Notebook
```bash
jupyter notebook notebooks/data_cleaning_and_feature_engineering.ipynb
```
