import os
import glob
import pandas as pd
import numpy as np


def load_raw_data(data_dir="data/raw"):
    files = sorted(glob.glob(os.path.join(data_dir, "atp_matches_*.csv")))
    if not files:
        files = sorted(glob.glob(os.path.join("..", data_dir, "atp_matches_*.csv")))
    if not files:
        files = sorted(glob.glob("atp_matches_*.csv"))

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["year"] = int(os.path.basename(f).split("_")[-1].replace(".csv", ""))
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def clean_data(df):
    df = df.drop_duplicates().reset_index(drop=True)
    df.columns = df.columns.str.strip().str.lower()

    numeric_cols = [
        "winner_rank", "loser_rank",
        "winner_rank_points", "loser_rank_points",
        "winner_age", "loser_age"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    categorical_cols = ["surface", "round", "tourney_name", "winner_name", "loser_name"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    required_cols = [
        "winner_id", "loser_id",
        "winner_rank", "loser_rank",
        "winner_rank_points", "loser_rank_points",
        "surface"
    ]
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    df["winner_age"] = df["winner_age"].fillna(df["winner_age"].median())
    df["loser_age"] = df["loser_age"].fillna(df["loser_age"].median())

    return df


def build_historical_features(df):
    df = df.sort_values(by=["year", "tourney_date"]).reset_index(drop=True)

    player_stats = {}
    h2h = {}
    rows = []

    def get_stats(pid):
        if pid not in player_stats:
            player_stats[pid] = {
                "matches": 0, "wins": 0,
                "surface_matches": {}, "surface_wins": {}
            }
        return player_stats[pid]

    def get_h2h(p1, p2):
        key = tuple(sorted([p1, p2]))
        if key not in h2h:
            h2h[key] = {p1: 0, p2: 0}
        return h2h[key].get(p1, 0)

    for _, row in df.iterrows():
        w_id = row["winner_id"]
        l_id = row["loser_id"]
        surface = row["surface"]

        if pd.isna(w_id) or pd.isna(l_id):
            continue

        w = get_stats(w_id)
        l = get_stats(l_id)

        w_win_rate = w["wins"] / w["matches"] if w["matches"] else 0.5
        l_win_rate = l["wins"] / l["matches"] if l["matches"] else 0.5

        w_sm = w["surface_matches"].get(surface, 0)
        w_sw = w["surface_wins"].get(surface, 0)
        l_sm = l["surface_matches"].get(surface, 0)
        l_sw = l["surface_wins"].get(surface, 0)

        w_surf_rate = w_sw / w_sm if w_sm else 0.5
        l_surf_rate = l_sw / l_sm if l_sm else 0.5

        w_h2h = get_h2h(w_id, l_id)
        l_h2h = get_h2h(l_id, w_id)

        # winner perspective
        rows.append({
            "player1": w_id, "player2": l_id, "surface": surface,
            "rank1": row["winner_rank"], "rank2": row["loser_rank"],
            "points1": row["winner_rank_points"], "points2": row["loser_rank_points"],
            "age1": row["winner_age"], "age2": row["loser_age"],
            "winrate1": w_win_rate, "winrate2": l_win_rate,
            "surfacerate1": w_surf_rate, "surfacerate2": l_surf_rate,
            "h2h1": w_h2h, "h2h2": l_h2h,
            "target": 1, "year": row["year"]
        })

        # loser perspective
        rows.append({
            "player1": l_id, "player2": w_id, "surface": surface,
            "rank1": row["loser_rank"], "rank2": row["winner_rank"],
            "points1": row["loser_rank_points"], "points2": row["winner_rank_points"],
            "age1": row["loser_age"], "age2": row["winner_age"],
            "winrate1": l_win_rate, "winrate2": w_win_rate,
            "surfacerate1": l_surf_rate, "surfacerate2": w_surf_rate,
            "h2h1": l_h2h, "h2h2": w_h2h,
            "target": 0, "year": row["year"]
        })

        # update rolling history after match
        w["matches"] += 1
        w["wins"] += 1
        l["matches"] += 1
        w["surface_matches"][surface] = w["surface_matches"].get(surface, 0) + 1
        w["surface_wins"][surface] = w["surface_wins"].get(surface, 0) + 1
        l["surface_matches"][surface] = l["surface_matches"].get(surface, 0) + 1

        key = tuple(sorted([w_id, l_id]))
        if key not in h2h:
            h2h[key] = {w_id: 0, l_id: 0}
        h2h[key][w_id] = h2h[key].get(w_id, 0) + 1

    return pd.DataFrame(rows)


def create_features(hist_df):
    feat = pd.DataFrame()
    feat["rank_diff"] = hist_df["rank1"] - hist_df["rank2"]
    feat["rank_points_diff"] = hist_df["points1"] - hist_df["points2"]
    feat["age_diff"] = hist_df["age1"] - hist_df["age2"]
    feat["win_rate_diff"] = hist_df["winrate1"] - hist_df["winrate2"]
    feat["surface_rate_diff"] = hist_df["surfacerate1"] - hist_df["surfacerate2"]
    feat["h2h_diff"] = hist_df["h2h1"] - hist_df["h2h2"]
    feat["surface"] = hist_df["surface"]
    feat["year"] = hist_df["year"]
    feat["target"] = hist_df["target"]

    numeric_cols = [
        "rank_diff", "rank_points_diff", "age_diff",
        "win_rate_diff", "surface_rate_diff", "h2h_diff"
    ]
    for col in numeric_cols:
        feat[col] = feat[col].fillna(feat[col].median())

    return feat


def main():
    print("Loading raw files...")
    raw_df = load_raw_data()
    print(f"Loaded {len(raw_df)} matches")

    print("Cleaning dataset...")
    cleaned = clean_data(raw_df)
    print(f"Cleaned dataset: {len(cleaned)} matches")

    print("Generating rolling features...")
    hist = build_historical_features(cleaned)

    print("Computing feature differentials...")
    features = create_features(hist)

    train = features[features["year"] < 2024].copy()
    test = features[features["year"] == 2024].copy()

    os.makedirs("data", exist_ok=True)
    train.to_csv("data/train_data.csv", index=False)
    test.to_csv("data/test_data.csv", index=False)
    cleaned.to_csv("data/cleaned_atp_data.csv", index=False)

    print(f"Train samples (2018-2023): {len(train)}")
    print(f"Test samples (2024):       {len(test)}")
    print("Saved files to data/")


if __name__ == "__main__":
    main()
