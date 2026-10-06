"""
ATP Tennis ML Project — Data Cleaning & Feature Engineering Pipeline
Prepares historical features and chronological train/test split for match prediction modeling.
"""

import os
import glob
import pandas as pd
import numpy as np


def run_pipeline(data_dir=".", output_dir="data"):
    """
    Executes the full pipeline:
    1. Ingests and combines ATP matches from 2018-2024.
    2. Cleans duplicates, standardizes columns, handles types and missing values.
    3. Calculates pre-match historical stats (win rate, surface rate, H2H) with zero leakage.
    4. Computes feature differentials and imputes missing values.
    5. Splits chronologically into Train (2018-2023) and Test (2024).
    """
    print("=" * 60)
    print("STEP 1: Load and combine the 7 ATP CSV files")
    print("=" * 60)
    
    files = sorted(glob.glob(os.path.join(data_dir, "atp_matches_*.csv")))
    if not files:
        files = sorted(glob.glob(os.path.join("..", "atp_matches_*.csv")))
        
    print(f"Files found ({len(files)}):")
    for f in files:
        print(f" - {f}")
        
    if len(files) != 7:
        print(f"WARNING: Expected 7 CSV files (2018-2024), found {len(files)}.")
        
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        year = int(os.path.basename(f).split("_")[-1].replace(".csv", ""))
        df["year"] = year
        dfs.append(df)
        
    data = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset shape: {data.shape}")

    print("\n" + "=" * 60)
    print("STEP 2: Clean the dataset")
    print("=" * 60)
    
    # Remove exact duplicate rows
    data = data.drop_duplicates().reset_index(drop=True)
    
    # Standardize column names
    data.columns = data.columns.str.strip().str.lower()
    
    # Convert important numeric columns
    numeric_cols = [
        "winner_rank", "loser_rank",
        "winner_rank_points", "loser_rank_points",
        "winner_age", "loser_age"
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            
    # Clean categorical columns
    categorical_cols = [
        "surface", "round", "tourney_name",
        "winner_name", "loser_name"
    ]
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype("string").str.strip()
            
    # Essential fields needed for features
    required_cols = [
        "winner_id", "loser_id",
        "winner_rank", "loser_rank",
        "winner_rank_points", "loser_rank_points",
        "surface"
    ]
    data = data.dropna(subset=required_cols).reset_index(drop=True)
    
    # Impute missing ages with median
    data["winner_age"] = data["winner_age"].fillna(data["winner_age"].median())
    data["loser_age"] = data["loser_age"].fillna(data["loser_age"].median())
    
    print(f"Cleaned dataset shape: {data.shape}")
    print("Null count in required columns:")
    for col in required_cols:
        print(f" - {col}: {data[col].isnull().sum()}")

    print("\n" + "=" * 60)
    print("STEP 4: Create chronological historical features (Zero Leakage)")
    print("=" * 60)
    
    history_data = data.copy()
    history_data = history_data.sort_values(by=["year", "tourney_date"]).reset_index(drop=True)
    
    player_stats = {}
    h2h = {}
    historical_rows = []
    
    def get_player_stats(player_id):
        if player_id not in player_stats:
            player_stats[player_id] = {
                "matches": 0,
                "wins": 0,
                "surface_matches": {},
                "surface_wins": {}
            }
        return player_stats[player_id]
        
    def get_h2h(player1, player2):
        key = tuple(sorted([player1, player2]))
        if key not in h2h:
            h2h[key] = {player1: 0, player2: 0}
        return h2h[key].get(player1, 0)

    for _, row in history_data.iterrows():
        winner = row["winner_id"]
        loser = row["loser_id"]
        surface = row["surface"]
        
        if pd.isna(winner) or pd.isna(loser):
            continue
            
        w = get_player_stats(winner)
        l = get_player_stats(loser)
        
        w_win_rate = w["wins"] / w["matches"] if w["matches"] else 0.5
        l_win_rate = l["wins"] / l["matches"] if l["matches"] else 0.5
        
        w_sm = w["surface_matches"].get(surface, 0)
        w_sw = w["surface_wins"].get(surface, 0)
        l_sm = l["surface_matches"].get(surface, 0)
        l_sw = l["surface_wins"].get(surface, 0)
        
        w_surface_rate = w_sw / w_sm if w_sm else 0.5
        l_surface_rate = l_sw / l_sm if l_sm else 0.5
        
        w_h2h = get_h2h(winner, loser)
        l_h2h = get_h2h(loser, winner)

        # Winner perspective: target = 1
        historical_rows.append({
            "player1": winner,
            "player2": loser,
            "surface": surface,
            "rank1": row["winner_rank"],
            "rank2": row["loser_rank"],
            "points1": row["winner_rank_points"],
            "points2": row["loser_rank_points"],
            "age1": row["winner_age"],
            "age2": row["loser_age"],
            "winrate1": w_win_rate,
            "winrate2": l_win_rate,
            "surfacerate1": w_surface_rate,
            "surfacerate2": l_surface_rate,
            "h2h1": w_h2h,
            "h2h2": l_h2h,
            "target": 1,
            "year": row["year"]
        })

        # Loser perspective: target = 0
        historical_rows.append({
            "player1": loser,
            "player2": winner,
            "surface": surface,
            "rank1": row["loser_rank"],
            "rank2": row["winner_rank"],
            "points1": row["loser_rank_points"],
            "points2": row["winner_rank_points"],
            "age1": row["loser_age"],
            "age2": row["winner_age"],
            "winrate1": l_win_rate,
            "winrate2": w_win_rate,
            "surfacerate1": l_surface_rate,
            "surfacerate2": w_surface_rate,
            "h2h1": l_h2h,
            "h2h2": w_h2h,
            "target": 0,
            "year": row["year"]
        })

        # Update historical statistics AFTER the match is processed
        w["matches"] += 1
        w["wins"] += 1
        l["matches"] += 1
        w["surface_matches"][surface] = w["surface_matches"].get(surface, 0) + 1
        w["surface_wins"][surface] = w["surface_wins"].get(surface, 0) + 1
        l["surface_matches"][surface] = l["surface_matches"].get(surface, 0) + 1
        
        key = tuple(sorted([winner, loser]))
        if key not in h2h:
            h2h[key] = {winner: 0, loser: 0}
        h2h[key][winner] = h2h[key].get(winner, 0) + 1

    historical_data = pd.DataFrame(historical_rows)
    print(f"Historical feature rows generated: {len(historical_data):,}")

    print("\n" + "=" * 60)
    print("STEP 5: Convert historical information into final ML features")
    print("=" * 60)
    
    improved_data = pd.DataFrame()
    improved_data["rank_diff"] = historical_data["rank1"] - historical_data["rank2"]
    improved_data["rank_points_diff"] = historical_data["points1"] - historical_data["points2"]
    improved_data["age_diff"] = historical_data["age1"] - historical_data["age2"]
    improved_data["win_rate_diff"] = historical_data["winrate1"] - historical_data["winrate2"]
    improved_data["surface_rate_diff"] = historical_data["surfacerate1"] - historical_data["surfacerate2"]
    improved_data["h2h_diff"] = historical_data["h2h1"] - historical_data["h2h2"]
    improved_data["surface"] = historical_data["surface"]
    improved_data["year"] = historical_data["year"]
    improved_data["target"] = historical_data["target"]

    numeric_features = [
        "rank_diff",
        "rank_points_diff",
        "age_diff",
        "win_rate_diff",
        "surface_rate_diff",
        "h2h_diff"
    ]
    for col in numeric_features:
        improved_data[col] = improved_data[col].fillna(improved_data[col].median())

    print(f"Final feature dataset shape: {improved_data.shape}")

    print("\n" + "=" * 60)
    print("STEP 6: Chronological train/test split — FINAL HANDOFF")
    print("=" * 60)
    
    train_data = improved_data[improved_data["year"] < 2024].copy()
    test_data = improved_data[improved_data["year"] == 2024].copy()

    print(f"Training samples (2018-2023): {len(train_data):,}")
    print(f"Testing samples (2024): {len(test_data):,}")
    print(f"Training years: {sorted(train_data['year'].unique())}")
    print(f"Testing years: {sorted(test_data['year'].unique())}")

    # Ensure output directories
    os.makedirs(output_dir, exist_ok=True)
    
    train_csv_path = os.path.join(output_dir, "train_data.csv")
    test_csv_path = os.path.join(output_dir, "test_data.csv")
    cleaned_csv_path = os.path.join(output_dir, "cleaned_atp_data.csv")
    
    train_data.to_csv(train_csv_path, index=False)
    test_data.to_csv(test_csv_path, index=False)
    data.to_csv(cleaned_csv_path, index=False)
    
    # Also save to root for direct path references
    train_data.to_csv("train_data.csv", index=False)
    test_data.to_csv("test_data.csv", index=False)
    data.to_csv("cleaned_atp_data.csv", index=False)

    print("\nFiles saved successfully:")
    print(f" - {train_csv_path}")
    print(f" - {test_csv_path}")
    print(f" - {cleaned_csv_path}")
    
    return train_data, test_data, improved_data


if __name__ == "__main__":
    run_pipeline()
