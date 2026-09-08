from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SRC_DIR / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "related_work_comparison.tex"


def main() -> None:
    table = r"""
\begin{table}[H]
\centering
\caption{Comparison of related IDS studies and the present work.}
\label{tab:related_work_comparison}
\resizebox{\textwidth}{!}{
\begin{tabular}{p{2.4cm}p{2.5cm}p{3.3cm}p{2.7cm}p{2.5cm}}
\hline
Study &
IDS approach &
Main feature perspective &
Domain variation considered &
Cross-domain evaluation \\
\hline

Kaveh et al.~\cite{REPLACE_KAVEH_KEY} &
Machine-learning-based IDS &
Primarily RPL routing and control-message features &
Attack variations and topology changes &
Yes; generalization across attack variations was examined \\

Violettas et al.~\cite{REPLACE_VIOLETTAS_KEY} &
Hybrid anomaly- and specification-based IDS &
Centralized RPL monitoring and attacker identification &
Multiple RPL attack conditions &
Not the main focus \\

Garcia Ribera et al.~\cite{REPLACE_GARCIA_KEY} &
Hybrid IDS &
Routing information together with CPU, TX, and RX overhead measurements &
Multiple RPL attacks &
Not systematically evaluated across the domain factors used here \\

Canbalaban et al.~\cite{REPLACE_CANBALABAN_KEY} &
Cross-layer IDS &
Routing-layer and link-layer features &
Different network and attack conditions &
Limited cross-domain analysis \\

This thesis &
LSTM-based IDS &
Baseline RPL features, RSSI, TX, and RX &
Four attacks, four network sizes, and three behavioral variants &
Yes; off-diagonal transfer across 48 domains \\
\hline
\end{tabular}
}
\end{table}
""".strip()

    OUTPUT_PATH.write_text(
        table,
        encoding="utf-8",
    )

    print(f"Generated:\n{OUTPUT_PATH}")
    print("\nReplace the four placeholder citation keys with the keys used in references.bib.")


if __name__ == "__main__":
    main()