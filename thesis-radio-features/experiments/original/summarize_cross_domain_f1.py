import pandas as pd
from pathlib import Path

CSV_PATH = Path("results") / "cross_test_summary.csv"
OUT_DIR = Path("results") / "cross_domain_summary"


def parse_domain(domain: str):
    parts = domain.split("_")
    if len(parts) < 3:
        return None, None, None
    node = parts[-2]
    version = parts[-1]
    attack = "_".join(parts[:-2])
    return attack, node, version


def add_domain_columns(df, col_name, prefix):
    parsed = df[col_name].apply(parse_domain)
    df[f"{prefix}_attack"] = parsed.apply(lambda x: x[0])
    df[f"{prefix}_node"] = parsed.apply(lambda x: x[1])
    df[f"{prefix}_version"] = parsed.apply(lambda x: x[2])
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    required = {"model_exp", "model_domain", "test_domain", "f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df["model_exp"] = df["model_exp"].astype(int)
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")

    df = add_domain_columns(df, "model_domain", "model")
    df = add_domain_columns(df, "test_domain", "test")

    
    df["is_diagonal"] = df["model_domain"] == df["test_domain"]

    
    df["same_attack"] = df["model_attack"] == df["test_attack"]
    df["same_node"] = df["model_node"] == df["test_node"]
    df["same_version"] = df["model_version"] == df["test_version"]

    
    def relation_label(row):
        if row["same_attack"] and row["same_node"] and row["same_version"]:
            return "same_attack_same_node_same_version"
        if row["same_attack"] and row["same_node"] and not row["same_version"]:
            return "same_attack_same_node_different_version"
        if row["same_attack"] and not row["same_node"] and row["same_version"]:
            return "same_attack_different_node_same_version"
        if row["same_attack"] and not row["same_node"] and not row["same_version"]:
            return "same_attack_different_node_different_version"
        if not row["same_attack"] and row["same_node"] and row["same_version"]:
            return "different_attack_same_node_same_version"
        if not row["same_attack"] and row["same_node"] and not row["same_version"]:
            return "different_attack_same_node_different_version"
        if not row["same_attack"] and not row["same_node"] and row["same_version"]:
            return "different_attack_different_node_same_version"
        return "different_attack_different_node_different_version"

    df["relation_group"] = df.apply(relation_label, axis=1)

    
    exp_mean = (
        df.groupby("model_exp", as_index=False)["f1"]
        .mean()
        .rename(columns={"f1": "mean_f1"})
        .sort_values("model_exp")
    )
    exp_mean.to_csv(OUT_DIR / "exp_mean_f1.csv", index=False)

    
    offdiag = df[~df["is_diagonal"]].copy()
    exp_offdiag_mean = (
        offdiag.groupby("model_exp", as_index=False)["f1"]
        .mean()
        .rename(columns={"f1": "offdiagonal_mean_f1"})
        .sort_values("model_exp")
    )
    exp_offdiag_mean.to_csv(OUT_DIR / "exp_offdiagonal_mean_f1.csv", index=False)

    
    
    attack_family_mean = (
        df.groupby(["model_exp", "test_attack"], as_index=False)["f1"]
        .mean()
        .rename(columns={"f1": "mean_f1"})
        .sort_values(["model_exp", "test_attack"])
    )
    attack_family_mean.to_csv(OUT_DIR / "attack_family_mean_f1.csv", index=False)

    
    condition_group_mean = (
        df.groupby(["model_exp", "relation_group"], as_index=False)["f1"]
        .mean()
        .rename(columns={"f1": "mean_f1"})
        .sort_values(["model_exp", "relation_group"])
    )
    condition_group_mean.to_csv(OUT_DIR / "condition_group_mean_f1.csv", index=False)

    
    same_attack_focus = df[df["same_attack"]].copy()

    def same_attack_focus_label(row):
        if row["same_node"] and row["same_version"]:
            return "same_attack_same_node_same_version"
        if (not row["same_node"]) and row["same_version"]:
            return "same_attack_different_node_same_version"
        if row["same_node"] and (not row["same_version"]):
            return "same_attack_same_node_different_version"
        return "same_attack_different_node_different_version"

    same_attack_focus["same_attack_focus_group"] = same_attack_focus.apply(same_attack_focus_label, axis=1)

    same_attack_focus_mean = (
        same_attack_focus.groupby(["model_exp", "same_attack_focus_group"], as_index=False)["f1"]
        .mean()
        .rename(columns={"f1": "mean_f1"})
        .sort_values(["model_exp", "same_attack_focus_group"])
    )
    same_attack_focus_mean.to_csv(OUT_DIR / "same_attack_focus_mean_f1.csv", index=False)

    print(f"Saved summaries to: {OUT_DIR}")


if __name__ == "__main__":
    main()