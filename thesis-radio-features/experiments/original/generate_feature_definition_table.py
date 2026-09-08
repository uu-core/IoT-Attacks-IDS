from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SRC_DIR / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "feature_definition_table.tex"


def main() -> None:
    table = r"""
\begin{table}[H]
\centering
\caption{Definitions and aggregation of the features used in the experiments.}
\label{tab:feature_definitions}
\resizebox{\textwidth}{!}{
\begin{tabular}{llll}
\hline
Feature group & Raw statistic & Description & Aggregated features \\
\hline
RPL baseline
& rank
& RPL rank of the node
& Mean and standard deviation \\

RPL baseline
& disr
& Number of received DIS messages
& Mean and standard deviation \\

RPL baseline
& diss
& Number of sent DIS messages
& Mean and standard deviation \\

RPL baseline
& dior
& Number of received DIO messages
& Mean and standard deviation \\

RPL baseline
& dios
& Number of sent DIO messages
& Mean and standard deviation \\

RPL baseline
& diar
& Number of received DAO messages
& Mean and standard deviation \\

RPL baseline
& tots
& Total number of sent RPL messages
& Mean and standard deviation \\

Radio
& RSSI
& Received signal strength indicator
& Mean and standard deviation \\

Radio activity
& TX
& Radio transmission activity
& Mean and standard deviation \\

Radio activity
& RX
& Radio reception activity
& Mean and standard deviation \\
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
    print("\nTable content:\n")
    print(table)


if __name__ == "__main__":
    main()