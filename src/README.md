
# 🧠 Continual Learning for IoT Intrusion Detection

This repository contains code for **domain-incremental continual learning (CL)** for Intrusion Detection Systems (IDS) in **RPL-based IoT networks**.  
The framework evaluates multiple CL methods — including **Synaptic Intelligence (SI)**, **Elastic Weight Consolidation (EWC)**, **Learning without Forgetting (LwF)**, **Replay**, and **Generative Replay (GR)** — under different **domain ordering scenarios**, such as Worst-to-Best, Best-to-Worst, and Toggle (adversarial).

---

## 📁 Project Structure

├── src/
│ ├── main.py # Entry point for launching experiments
│ ├── models.py # LSTM-based IDS models
│ ├── utils.py # Data loading, preprocessing, metric utilities
│ ├── train_WCL_w2b_b2w_togg.py # Baseline: sequential fine-tuning under different orderings
│ ├── train_CL_SI.py # Synaptic Intelligence training
│ ├── train_CL_EWC.py # Elastic Weight Consolidation training
│ ├── train_CL_LWF.py # Learning without Forgetting training
│ ├── train_CL_genreplay.py # Generative Replay training
│
├── Attack_data/ # <-- Datasets are stored OUTSIDE src/
│ ├── Domain_1/
│ │ ├── data_1.csv
│ │ ├── data_2.csv
│ │ └── ...
│ ├── Domain_2/
│ │ ├── data_1.csv
│ │ ├── data_2.csv
│ │ └── ...
│ └── ...
│
└── README.md


> ⚠️ **Important:** The attack datasets are stored **outside** the `src/` directory. Each domain corresponds to a specific attack–variant–network size combination, and contains one or more CSV files.

---

## 🧰 Requirements

- Python 3.8+
- PyTorch
- NumPy
- tqdm
- wandb *(optional)*

Install dependencies:

```bash
pip install -r requirements.txt


Dataset Format

The dataset is structured into domains, each representing a unique combination of attack type, behavioral variant, and network size.
Within each domain, multiple CSV files contain time-windowed feature logs of IoT traffic.

Attack_data/
├── Domain_1/
│   ├── data_1.csv
│   ├── data_2.csv
│   └── ...
├── Domain_2/
│   ├── data_1.csv
│   ├── data_2.csv
│   └── ...
└── ...

Features: 14 per record (control message counts, rank changes, packet stats, etc.)

Labels: 0 (benign) or 1 (attack)

Data is pre-windowed before training using sliding windows.

You can specify the path to Attack_data when running the training scripts using the --data_root argument.


Running Experiments

All training scripts are located in src/. Each script corresponds to a specific continual learning strategy.

Baseline (W/O CL)
python src/train_WCL_w2b_b2w_togg.py --data_root /path/to/Attack_data --scenario w2b

Synaptic Intelligence
python src/train_CL_SI.py --data_root /path/to/Attack_data

Elastic Weight Consolidation
python src/train_CL_EWC.py --data_root /path/to/Attack_data

Learning without Forgetting
python src/train_CL_LWF.py --data_root /path/to/Attack_data

Generative Replay
python src/train_CL_genreplay.py --data_root /path/to/Attack_data

Hyperparameters

All scripts support various hyperparameters such as learning rate, epochs, method-specific weights (e.g., si_c, ewc_lambda, alpha, replay buffer size, etc.).
Please refer to the Hyperparameters section in the code (utils.py and training scripts) to tune them as needed.