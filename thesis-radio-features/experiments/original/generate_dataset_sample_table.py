from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths and parameters
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent


DATA_ROOT = PROJECT_ROOT / "attack_data"


MAPPING_CANDIDATES = [
    SRC_DIR / "domain_details.xlsx",
    PROJECT_ROOT / "domain_details.xlsx",
]

OUTPUT_DIR = SRC_DIR / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_OUTPUT = OUTPUT_DIR / "dataset_sample_counts_by_domain.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "dataset_sample_summary.csv"
LATEX_OUTPUT = OUTPUT_DIR / "dataset_sample_summary.tex"

SEQUENCE_LENGTH = 10
STEP = 1


# ============================================================
# Allowed domains
# ============================================================

ATTACK_LABELS = {
    "blackhole": "Blackhole",
    "dis_flooding": "DIS Flooding",
    "local_repair": "Local Repair",
    "worst_parent": "Worst Parent",
}

ATTACK_ORDER = [
    "Blackhole",
    "DIS Flooding",
    "Local Repair",
    "Worst Parent",
]

VARIANT_LABELS = {
    "base": "base",
    "oo": "oo",
    "gc": "gc",
}

DOMAIN_PATTERN = re.compile(
    r"^(blackhole|dis_flooding|local_repair|worst_parent)"
    r"_(5|10|15|20)_(base|oo|gc)$",
    re.IGNORECASE,
)

RAW_DOMAIN_PATTERN = re.compile(
    r"^domain[\s_-]*0*(\d+)$",
    re.IGNORECASE,
)

LABEL_COLUMN_CANDIDATES = [
    "label",
    "labels",
    "attack_label",
    "is_attack",
    "target",
    "class",
    "y",
    "attack",
]


# ============================================================
# Normalization helpers
# ============================================================

def normalize_text(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("–", "-")
        .replace("—", "-")
    )


def normalize_attack(value: object) -> str | None:
    text = normalize_text(value)

    normalized = (
        text
        .replace("-", "_")
        .replace(" ", "_")
    )

    aliases = {
        "blackhole": "blackhole",
        "black_hole": "blackhole",
        "dis_flooding": "dis_flooding",
        "disflooding": "dis_flooding",
        "local_repair": "local_repair",
        "localrepair": "local_repair",
        "worst_parent": "worst_parent",
        "worstparent": "worst_parent",
    }

    if normalized in aliases:
        return aliases[normalized]

    for alias, attack_key in aliases.items():
        if alias in normalized:
            return attack_key

    return None


def normalize_variant(value: object) -> str | None:
    text = (
        normalize_text(value)
        .replace("-", "_")
        .replace(" ", "_")
    )

    aliases = {
        "base": "base",
        "basic": "base",
        "oo": "oo",
        "on_off": "oo",
        "onoff": "oo",
        "gc": "gc",
        "gradual": "gc",
        "gradual_change": "gc",
        "gradualchange": "gc",
    }

    if text in aliases:
        return aliases[text]

    return None


def extract_node_count(value: object) -> int | None:
    text = normalize_text(value)

    # Exact numeric cell
    try:
        numeric_value = float(text)

        if numeric_value.is_integer():
            integer_value = int(numeric_value)

            if integer_value in {5, 10, 15, 20}:
                return integer_value
    except ValueError:
        pass

    # Text such as "15 nodes"
    match = re.search(
        r"\b(5|10|15|20)\s*(?:nodes?)?\b",
        text,
    )

    if match:
        return int(match.group(1))

    return None


def normalize_raw_domain(value: object) -> str | None:
    text = normalize_text(value)

    match = RAW_DOMAIN_PATTERN.fullmatch(text)

    if not match:
        return None

    number = int(match.group(1))

    return f"domain{number:02d}"


def normalize_canonical_domain(
    value: object,
) -> str | None:
    text = (
        normalize_text(value)
        .replace("-", "_")
        .replace(" ", "_")
    )

    if DOMAIN_PATTERN.fullmatch(text):
        return text

    return None


def parse_canonical_domain(
    domain_name: str,
) -> tuple[str, int, str]:
    match = DOMAIN_PATTERN.fullmatch(
        domain_name
    )

    if match is None:
        raise ValueError(
            f"Invalid canonical domain name: {domain_name}"
        )

    attack_key, node_text, variant_key = (
        match.groups()
    )

    return (
        attack_key,
        int(node_text),
        variant_key,
    )


# ============================================================
# Domain mapping
# ============================================================

def find_mapping_file() -> Path:
    for candidate in MAPPING_CANDIDATES:
        if candidate.exists():
            return candidate

    checked_paths = "\n".join(
        str(path)
        for path in MAPPING_CANDIDATES
    )

    raise FileNotFoundError(
        "domain_details.xlsx was not found.\n"
        f"Checked:\n{checked_paths}"
    )


def load_domain_mapping() -> dict[
    tuple[str, str],
    str,
]:
    """
    Returns mappings such as:

        ("blackhole", "domain01")
            -> "blackhole_5_base"

    It supports either:
    1. a readable domain name already written in one cell, or
    2. separate attack, node-count and variant columns.

    It also supports separate Excel sheets named after attacks.
    """
    mapping_file = find_mapping_file()

    workbook = pd.read_excel(
        mapping_file,
        sheet_name=None,
        dtype=str,
    )

    mapping: dict[
        tuple[str, str],
        str,
    ] = {}

    for sheet_name, dataframe in workbook.items():
        dataframe = dataframe.fillna("")

        sheet_attack = normalize_attack(
            sheet_name
        )

        for _, row in dataframe.iterrows():
            values = [
                value
                for value in row.tolist()
                if str(value).strip()
            ]

            if not values:
                continue

            raw_domain = None
            canonical_domain = None
            attack_key = None
            node_count = None
            variant_key = None

            for value in values:
                if raw_domain is None:
                    raw_domain = normalize_raw_domain(
                        value
                    )

                if canonical_domain is None:
                    canonical_domain = (
                        normalize_canonical_domain(
                            value
                        )
                    )

                if attack_key is None:
                    attack_key = normalize_attack(
                        value
                    )

                if node_count is None:
                    node_count = extract_node_count(
                        value
                    )

                if variant_key is None:
                    variant_key = normalize_variant(
                        value
                    )

            # If the complete readable domain is present,
            # extract attack/node/variant from it.
            if canonical_domain is not None:
                (
                    canonical_attack,
                    canonical_nodes,
                    canonical_variant,
                ) = parse_canonical_domain(
                    canonical_domain
                )

                attack_key = canonical_attack
                node_count = canonical_nodes
                variant_key = canonical_variant

            # Sheet name may provide the attack type.
            if attack_key is None:
                attack_key = sheet_attack

            if (
                canonical_domain is None
                and attack_key is not None
                and node_count is not None
                and variant_key is not None
            ):
                canonical_domain = (
                    f"{attack_key}_"
                    f"{node_count}_"
                    f"{variant_key}"
                )

            if (
                raw_domain is not None
                and attack_key in ATTACK_LABELS
                and canonical_domain is not None
            ):
                mapping[
                    (attack_key, raw_domain)
                ] = canonical_domain

    if not mapping:
        raise RuntimeError(
            "No domain mappings could be read from:\n"
            f"{mapping_file}\n\n"
            "The spreadsheet must contain domain folders "
            "such as domain01 together with attack type, "
            "node count and variant information."
        )

    print(
        f"Loaded {len(mapping)} domain mappings from:\n"
        f"{mapping_file}"
    )

    return mapping


# ============================================================
# Label handling
# ============================================================

def find_label_column(
    dataframe: pd.DataFrame,
    csv_path: Path,
) -> str:
    normalized_columns = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in normalized_columns:
            return normalized_columns[candidate]

    raise KeyError(
        f"No label column found in:\n{csv_path}\n\n"
        f"Available columns:\n"
        f"{list(dataframe.columns)}\n\n"
        "Add the actual label-column name to "
        "LABEL_COLUMN_CANDIDATES."
    )


def convert_labels_to_binary(
    series: pd.Series,
    csv_path: Path,
) -> np.ndarray:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().all():
        return (
            numeric.to_numpy() > 0
        ).astype(int)

    benign_values = {
        "0",
        "benign",
        "normal",
        "false",
        "no",
        "inactive",
    }

    attack_values = {
        "1",
        "attack",
        "malicious",
        "true",
        "yes",
        "active",
    }

    binary_labels: list[int] = []

    for value in series:
        normalized_value = normalize_text(
            value
        )

        if normalized_value in benign_values:
            binary_labels.append(0)

        elif normalized_value in attack_values:
            binary_labels.append(1)

        else:
            raise ValueError(
                f"Unknown label value '{value}' in:\n"
                f"{csv_path}"
            )

    return np.asarray(
        binary_labels,
        dtype=int,
    )


# ============================================================
# Sequence counting
# ============================================================

def count_sequences(
    labels: np.ndarray,
) -> tuple[int, int, int]:
    """
    Reproduces the sequence construction and phase-based
    labeling implemented in utils.py.
    """
    number_of_rows = len(labels)

    # utils.py uses:
    # range(len(df_feat) - sequence_length)
    total_sequences = max(
        number_of_rows - SEQUENCE_LENGTH,
        0,
    )

    if total_sequences == 0:
        return 0, 0, 0

    attack_indices = np.flatnonzero(
        labels == 1
    )

    if len(attack_indices) == 0:
        return (
            total_sequences,
            0,
            total_sequences,
        )

    first_attack_index = int(
        attack_indices[0]
    )

    # Exact implementation in utils.py:
    # start_attack = first_attack_index - sequence_length
    start_attack = max(
        0,
        first_attack_index - SEQUENCE_LENGTH,
    )

    benign_sequences = min(
        start_attack,
        total_sequences,
    )

    attack_sequences = (
        total_sequences - benign_sequences
    )

    return (
        benign_sequences,
        attack_sequences,
        total_sequences,
    )

# ============================================================
# Load the actual folder structure
# ============================================================

def load_dataset_counts() -> pd.DataFrame:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Data directory not found:\n"
            f"{DATA_ROOT}"
        )

    domain_mapping = load_domain_mapping()

    rows: list[dict[str, object]] = []
    missing_mappings: set[
        tuple[str, str]
    ] = set()

    # Top-level folder is the attack type.
    for attack_dir in sorted(
        DATA_ROOT.iterdir()
    ):
        if not attack_dir.is_dir():
            continue

        attack_key = normalize_attack(
            attack_dir.name
        )

        # Automatically excludes failing_node.
        if attack_key not in ATTACK_LABELS:
            print(
                f"Skipped excluded folder: "
                f"{attack_dir.name}"
            )
            continue

        # Second-level folder is domain01, domain02, etc.
        for raw_domain_dir in sorted(
            attack_dir.iterdir()
        ):
            if not raw_domain_dir.is_dir():
                continue

            raw_domain = normalize_raw_domain(
                raw_domain_dir.name
            )

            if raw_domain is None:
                print(
                    f"Skipped unrecognized folder:\n"
                    f"{raw_domain_dir}"
                )
                continue

            mapping_key = (
                attack_key,
                raw_domain,
            )

            canonical_domain = (
                domain_mapping.get(
                    mapping_key
                )
            )

            if canonical_domain is None:
                missing_mappings.add(
                    mapping_key
                )
                continue

            (
                canonical_attack,
                nodes,
                variant,
            ) = parse_canonical_domain(
                canonical_domain
            )

            csv_paths = sorted(
                raw_domain_dir.glob("*.csv")
            )

            if not csv_paths:
                print(
                    f"No CSV files found in:\n"
                    f"{raw_domain_dir}"
                )
                continue

            for csv_path in csv_paths:
                dataframe = pd.read_csv(
                    csv_path
                )

                label_column = (
                    find_label_column(
                        dataframe,
                        csv_path,
                    )
                )

                labels = (
                    convert_labels_to_binary(
                        dataframe[
                            label_column
                        ],
                        csv_path,
                    )
                )

                (
                    benign_sequences,
                    attack_sequences,
                    total_sequences,
                ) = count_sequences(labels)

                rows.append(
                    {
                        "attack": ATTACK_LABELS[
                            canonical_attack
                        ],
                        "domain": canonical_domain,
                        "raw_domain": raw_domain,
                        "nodes": nodes,
                        "variant": variant,
                        "run_file": csv_path.name,
                        "raw_rows": len(
                            dataframe
                        ),
                        "benign_sequences":
                            benign_sequences,
                        "attack_sequences":
                            attack_sequences,
                        "total_sequences":
                            total_sequences,
                    }
                )

    if missing_mappings:
        missing_text = "\n".join(
            f"{attack}/{domain}"
            for attack, domain
            in sorted(missing_mappings)
        )

        raise RuntimeError(
            "The following folders were not found in "
            "domain_details.xlsx:\n"
            f"{missing_text}"
        )

    if not rows:
        raise RuntimeError(
            "No valid CSV files were loaded."
        )

    detailed_dataframe = pd.DataFrame(
        rows
    )

    detailed_dataframe.to_csv(
        DETAIL_OUTPUT,
        index=False,
    )

    return detailed_dataframe


# ============================================================
# Summary and LaTeX table
# ============================================================

def create_summary(
    detailed_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        detailed_dataframe
        .groupby(
            "attack",
            observed=True,
        )
        .agg(
            domains=("domain", "nunique"),
            runs=("run_file", "count"),
            raw_rows=("raw_rows", "sum"),
            benign_sequences=(
                "benign_sequences",
                "sum",
            ),
            attack_sequences=(
                "attack_sequences",
                "sum",
            ),
            total_sequences=(
                "total_sequences",
                "sum",
            ),
        )
        .reset_index()
    )

    summary["attack"] = pd.Categorical(
        summary["attack"],
        categories=ATTACK_ORDER,
        ordered=True,
    )

    summary = (
        summary
        .sort_values("attack")
        .reset_index(drop=True)
    )

    total_row = pd.DataFrame(
        [
            {
                "attack": "Total",
                "domains":
                    summary["domains"].sum(),
                "runs":
                    summary["runs"].sum(),
                "raw_rows":
                    summary["raw_rows"].sum(),
                "benign_sequences":
                    summary[
                        "benign_sequences"
                    ].sum(),
                "attack_sequences":
                    summary[
                        "attack_sequences"
                    ].sum(),
                "total_sequences":
                    summary[
                        "total_sequences"
                    ].sum(),
            }
        ]
    )

    summary = pd.concat(
        [summary, total_row],
        ignore_index=True,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    return summary


def format_integer(
    value: int | float,
) -> str:
    return f"{int(value):,}"


def generate_latex_table(
    summary: pd.DataFrame,
) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        (
            r"\caption{Dataset size and sequence-class "
            r"distribution by attack scenario.}"
        ),
        r"\label{tab:dataset_sample_summary}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        (
            r"Attack scenario & Domains & Runs "
            r"& Raw rows & Benign sequences "
            r"& Attack sequences & Total sequences \\"
        ),
        r"\hline",
    ]

    for _, row in summary.iterrows():
        attack_name = str(
            row["attack"]
        )

        if attack_name == "Total":
            attack_name = r"\textbf{Total}"

        lines.append(
            f"{attack_name} "
            f"& {format_integer(row['domains'])} "
            f"& {format_integer(row['runs'])} "
            f"& {format_integer(row['raw_rows'])} "
            f"& {format_integer(row['benign_sequences'])} "
            f"& {format_integer(row['attack_sequences'])} "
            f"& {format_integer(row['total_sequences'])} "
            r"\\"
        )

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
        ]
    )

    LATEX_OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\nLaTeX table:\n")
    print("\n".join(lines))


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        f"Reading CSV files from:\n"
        f"{DATA_ROOT}\n"
    )

    detailed_dataframe = (
        load_dataset_counts()
    )

    summary = create_summary(
        detailed_dataframe
    )

    generate_latex_table(
        summary
    )

    print("\nSummary:")
    print(
        summary.to_string(index=False)
    )

    print(
        f"\nDetailed CSV:\n{DETAIL_OUTPUT}"
    )

    print(
        f"\nSummary CSV:\n{SUMMARY_OUTPUT}"
    )

    print(
        f"\nLaTeX table:\n{LATEX_OUTPUT}"
    )

    total_runs = int(
        summary.loc[
            summary["attack"] == "Total",
            "runs",
        ].iloc[0]
    )

    if total_runs != 960:
        print(
            f"\nWARNING: Found {total_runs} runs; "
            "960 were expected."
        )


if __name__ == "__main__":
    main()