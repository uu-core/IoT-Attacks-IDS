# 🧠 Quantifying Catastrophic Forgetting in IoT Intrusion Detection Systems

This repository contains code for **domain-incremental continual learning (CL)** for Intrusion Detection Systems (IDS) in **RPL-based IoT networks**.  
The framework evaluates multiple CL methods — including **Synaptic Intelligence (SI)**, **Elastic Weight Consolidation (EWC)**, **Learning without Forgetting (LwF)**, **Replay**, and **Generative Replay (GR)** — under different **domain ordering scenarios**, such as Worst-to-Best, Best-to-Worst, and Toggle (adversarial).

---

## 📁 Project Structure

```
.
├── src/
│   ├── main.py                      # Entry point for launching experiments
│   ├── models.py                    # LSTM-based IDS models
│   ├── utils.py                     # Data loading, preprocessing, metric utilities
│   ├── train_WCL_w2b_b2w_togg.py    # Baseline: sequential fine-tuning under different orderings
│   ├── train_CL_SI.py               # Synaptic Intelligence training
│   ├── train_CL_EWC.py              # Elastic Weight Consolidation training
│   ├── train_CL_LWF.py              # Learning without Forgetting training
│   ├── train_CL_genreplay.py       # Generative Replay training
│   │
│   ├── Attack_data/
│   │   ├── Domain_1/
│   │   │   ├── data_1.csv
│   │   │   ├── data_2.csv
│   │   │   └── ...
│   │   ├── Domain_2/
│   │   │   ├── data_1.csv
│   │   │   ├── data_2.csv
│   │   │   └── ...
│   │   └── ...
│
└── README.md
```

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
```

---

## 📊 Dataset Format

The dataset is located in **`src/Attack_data/`**.  
It is organized into **domains**, each representing a unique combination of attack type, behavioral variant, and network size.  
Within each domain, multiple CSV files contain time-windowed feature logs of IoT traffic.

```
src/Attack_data/
├── Domain_1/
│   ├── data_1.csv
│   ├── data_2.csv
│   └── ...
├── Domain_2/
│   ├── data_1.csv
│   ├── data_2.csv
│   └── ...
└── ...
```

- **Features:** 14 per record (control message counts, rank changes, packet stats, etc.)  
- **Labels:** `0` (benign) or `1` (attack)  
- Data is pre-windowed before training using sliding windows.

The code automatically loads data from this folder — **no manual path specification is needed**.

---

## 🚀 Running Experiments

All training scripts are located in `src/`. Each script corresponds to a specific continual learning strategy.

### Baseline (W/O CL)
```bash
python src/train_WCL_w2b_b2w_togg.py --scenario w2b
```

### Synaptic Intelligence
```bash
python src/train_CL_SI.py
```

### Elastic Weight Consolidation
```bash
python src/train_CL_EWC.py
```

### Learning without Forgetting
```bash
python src/train_CL_LWF.py
```

### Generative Replay
```bash
python src/train_CL_genreplay.py
```

---

## ⚙️ Hyperparameters (used in our experiments)

**Run context**
- Project: `attack_CL`
- Entity: `sourasb05`
- Algorithm: `Replay`
- Scenario: `w2b`
- Architecture: `LSTM`

### Global / shared
- Learning rate: `0.001`
- Epochs: `100`
- Batch size: `256`
- Early stopping patience: `50`
- Window size: `10`
- Step size: `3`
- Weight decay: `0.0001`
- Forgetting threshold (reporting): `0.01`
- Seed: (default in code, typically `42`)

### Model
- Input size: `140`
- Hidden size: `10`
- Output size: `2`
- LSTM layers: `1`
- Dropout: `0.05`
- Bidirectional: `False` (default unless set elsewhere)

### Method-specific (Replay) — **active**
- Total replay capacity (`memory_size`): `4000`
- Per-domain cap (`replay_per_domain_cap`): **`250`**
  - ⚠️ Note: Code hard-codes `250`. Your env had `PER_DOMAIN_CAP=300`, but this value is **ignored** by the current call.
- Replay batch size (`replay_batch_size`): `128`
- Replay ratio (`replay_ratio`): `0.5`
- Replay seen-only (`replay_seen_only`): `True`

### Method-specific knobs (present in args but **not used** in this Replay run)
- **LwF**: `alpha=1.0`, `temperature=4.0`, `enc_lr_scale=0.5`, `warmup_epochs=10`
- **SI**: `si_c`, `si_xi`, schedules (unused here)
- **EWC**: `ewc_lambda`, λ schedules, Fisher options (unused here)
- **GR**: VAE and distillation parameters (unused here)
---

## 📝 Outputs

- Each experiment produces a JSON file containing per-domain metrics:
  - Performance (F1/AUC)
  - Stability (BWT)
  - Plasticity
  - Efficiency
- These logs can be used for visualization, comparison across CL methods, or statistical analysis.

---

## 🧠 Citation

If you use this code or dataset in your research, please cite:

```

```
