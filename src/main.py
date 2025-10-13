import utils as utils
import models as models
import train_CL as train_WCL
import train_WCL_w2b_b2w_togg as train_WCL_w2b_b2w_togg
import train_CL_EWC_w2b_b2w_togg as EWC_w2b_b2w_togg
import tdim_replay as tdim_replay
import train_WCL_b2w as train_WCL_b2w
import train_CL_SI as train_si
import train_CL_EWC as train_ewc
import train_CL_EWC_ZS as train_ewc_zs
import train_CL_genreplay as train_genreplay
import train_CL_LWF as train_lwf
from utils import cluster_domains  
import numpy as np
from tqdm import trange
import torch
import os
import random
import wandb
import sys
import logging
import datetime
from types import SimpleNamespace

class NoOpWandbRun:
    def __init__(self):
        # config.update(...) should not crash
        self.config = SimpleNamespace(update=lambda *a, **k: None)
        # summary acts like a dict
        self.summary = {}
    def log(self, *a, **k): pass
    def watch(self, *a, **k): pass
    def define_metric(self, *a, **k): pass
    def finish(self): pass

def main():
    args = utils.parse_args()
    if not args.use_wandb:
        os.environ["WANDB_DISABLED"] = "true"  # prevents accidental init

    #gpu = args.gpu
    #device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    #print("device :",device)
    # ----------------------------
    # Wandb Setup
    # ----------------------------
    # Start a new wandb run to track this script.
    if args.use_wandb:
        run_wandb = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity=args.entity,
        # Set the wandb project where this run will be logged.
        project=args.project,
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": args.learning_rate,
            "architecture": args.architecture,
            "dataset": "vinnova_attack_dataset",
            "epochs": args.epochs,
            "algorithm": args.algorithm,
            "scenario": args.scenario,
            "exp_no": args.exp_no,
            "window_size": args.window_size,
            "step_size": args.step_size,
            "input_size": args.input_size,
            "hidden_size": args.hidden_size,
            "output_size": args.output_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "bidirectional": args.bidirectional
        },
    )
    else:
        run_wandb = NoOpWandbRun()
    # ----------------------------
    # 0. Device Setup (MPS/CUDA/CPU)
    # ----------------------------
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using Apple MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    current_directory = os.getcwd()
    
    # print(current_directory)

    algorithm = args.algorithm  # "SI/EWC/WCL/Generative_Replay/" 
    scenario = args.scenario    # (randoem/b2w/w2b/clustered/toggle)
    architecture = args.architecture # LSTM/BiLSTM/LSTM_Attention/BiLSTM_Attention/LSTM_Attention_adapter
    exp_no = args.exp_no
    
    # === Setup Logging ===
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(current_directory, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"{architecture}_{algorithm}_{scenario}_log_{timestamp}.log")

    logging.basicConfig(
        filename=log_filename,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Also print logs to the console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info(f"Initialized logging for: {architecture} | {algorithm} | {scenario}  | Experiment No: {args.exp_no}")
    
    logging.info(f"Logs will be saved to: {log_filename}")
    
    domains_path = current_directory + '/data/attack_data'

    domains = utils.create_domains(domains_path)

    train_domains_loader = {}
    test_domains_loader = {}
    full_domains_loader = {}

    for key, files in domains.items():
        train_domains_loader[key], test_domains_loader[key] = utils.load_data(domains_path, key, files, window_size=args.window_size, step_size=args.step_size, batch_size=args.batch_size)
          
    
    if architecture == "LSTM":
        model = models.LSTMClassifier(input_dim=args.input_size, hidden_dim=args.hidden_size, output_dim=args.output_size, num_layers=args.num_layers, fc_hidden_dim=10).to(device)
    elif architecture == "BiLSTM":
        model = models.LSTMModel(input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional).to(device)
    elif architecture == "LSTM_Attention":
        model = models.LSTMModelWithAttention(input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional).to(device)
    elif architecture == "BiLSTM_Attention":
        model = models.LSTMModelWithAttention(input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional).to(device)
    elif architecture == "LSTM_Attention_adapter":
        model = models.LSTMWithAdapterClassifier(input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size, num_domains=len(train_domains_loader), num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional).to(device)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    logging.info(f" Experiment No: {exp_no} | Algorithm: {algorithm} | Scenario: {scenario}")
    logging.info(f"Model architecture: {architecture} | Input size: {args.input_size} | Hidden size: {args.hidden_size} | Output size: {args.output_size} | Num layers: {args.num_layers} | Dropout: {args.dropout} | Bidirectional: {args.bidirectional}")
    
    if scenario == "random":
        train_domain_order = ["blackhole_var10_base", "blackhole_var10_dec", "blackhole_var10_oo", "blackhole_var15_base",
        "blackhole_var15_dec", "blackhole_var15_oo", "blackhole_var20_base", "blackhole_var20_dec", "blackhole_var20_oo", "blackhole_var5_base",
        "blackhole_var5_dec", "blackhole_var5_oo", "disflooding_var10_base", "disflooding_var10_dec", "disflooding_var10_oo", "disflooding_var15_base",
        "disflooding_var15_dec", "disflooding_var15_oo", "disflooding_var20_base", "disflooding_var20_dec", "disflooding_var20_oo", "disflooding_var5_base",
        "disflooding_var5_dec", "disflooding_var5_oo", "localrepair_var10_base", "localrepair_var10_dec", "localrepair_var10_oo", "localrepair_var15_base",
        "localrepair_var15_dec", "localrepair_var15_oo", "localrepair_var20_base", "localrepair_var20_dec", "localrepair_var20_oo", "localrepair_var5_base",
        "localrepair_var5_dec", "localrepair_var5_oo", "worstparent_var10_base", "worstparent_var10_dec", "worstparent_var10_oo", "worstparent_var15_base",
        "worstparent_var15_dec", "worstparent_var15_oo", "worstparent_var20_base", "worstparent_var20_dec", "worstparent_var20_oo", "worstparent_var5_base",
        "worstparent_var5_dec", "worstparent_var5_oo"]

    elif scenario == "w2b":
        train_domain_order = [ "localrepair_var5_oo", "blackhole_var10_oo", "blackhole_var5_oo", "blackhole_var15_dec", "blackhole_var20_oo", "worstparent_var5_oo",
        "worstparent_var10_oo", "blackhole_var15_oo", "blackhole_var10_dec", "worstparent_var5_base", "worstparent_var10_base", "localrepair_var15_oo",
        "localrepair_var10_base", "worstparent_var20_dec", "localrepair_var20_oo", "worstparent_var5_dec", "worstparent_var15_oo", "localrepair_var10_oo",
        "localrepair_var15_dec", "blackhole_var20_base", "worstparent_var20_oo", "worstparent_var10_dec", "worstparent_var20_base", "worstparent_var15_base",
        "disflooding_var20_dec", "blackhole_var20_dec", "localrepair_var5_dec", "localrepair_var20_base", "disflooding_var20_base", "blackhole_var15_base",
        "blackhole_var10_base", "localrepair_var5_base", "localrepair_var10_dec", "blackhole_var5_base", "localrepair_var15_base", "disflooding_var20_oo",
        "blackhole_var5_dec", "localrepair_var20_dec", "disflooding_var15_oo", "worstparent_var15_dec", "disflooding_var5_base", "disflooding_var10_base",
        "disflooding_var5_dec", "disflooding_var5_oo", "disflooding_var15_dec", "disflooding_var10_dec", "disflooding_var15_base", "disflooding_var10_oo" ]
    elif scenario == "b2w":
        train_domain_order = [ "localrepair_var5_oo", "disflooding_var20_base", "disflooding_var20_oo", "disflooding_var15_oo", "disflooding_var15_base",
        "disflooding_var20_dec", "disflooding_var15_dec", "disflooding_var10_base", "disflooding_var10_dec", "disflooding_var10_oo", "disflooding_var5_oo",
        "disflooding_var5_base", "localrepair_var15_dec", "disflooding_var5_dec", "localrepair_var10_dec", "localrepair_var5_dec", "localrepair_var15_oo",
        "localrepair_var20_dec", "localrepair_var5_base", "localrepair_var20_base", "localrepair_var10_base", "localrepair_var15_base", "worstparent_var20_base",
        "worstparent_var20_oo", "worstparent_var15_base", "blackhole_var20_dec", "blackhole_var10_dec", "blackhole_var15_base", "localrepair_var10_oo",
        "localrepair_var20_oo", "worstparent_var15_dec", "worstparent_var10_base", "worstparent_var10_oo", "worstparent_var15_oo", "blackhole_var15_dec",
        "blackhole_var20_base", "worstparent_var20_dec", "worstparent_var5_base", "worstparent_var10_dec", "blackhole_var5_dec", "blackhole_var5_base",
        "blackhole_var10_base", "blackhole_var10_oo", "blackhole_var15_oo", "blackhole_var20_oo", "blackhole_var5_oo", "worstparent_var5_dec", "worstparent_var5_oo"
        ]
    elif scenario == "toggle":
        train_domain_order = [ "localrepair_var5_oo", "blackhole_var10_oo", "disflooding_var10_dec", "blackhole_var5_oo", "disflooding_var10_oo",
        "blackhole_var15_dec", "disflooding_var15_dec", "blackhole_var15_oo", "disflooding_var20_oo", "blackhole_var20_dec", "disflooding_var15_base", 
        "blackhole_var15_base", "blackhole_var10_dec", "blackhole_var20_oo", "disflooding_var20_dec", "localrepair_var20_base", "localrepair_var15_base",
        "worstparent_var20_dec", "worstparent_var15_dec", "localrepair_var10_oo", "localrepair_var20_dec", "localrepair_var10_base", "worstparent_var20_oo",
        "worstparent_var10_base", "worstparent_var15_oo", "worstparent_var5_base", "worstparent_var15_base", "localrepair_var20_oo", "localrepair_var15_oo",
        "worstparent_var5_dec", "worstparent_var10_dec", "worstparent_var5_oo", "disflooding_var20_base", "blackhole_var20_base", "worstparent_var20_base",
        "worstparent_var10_oo", "disflooding_var15_oo", "blackhole_var10_base", "disflooding_var5_base", "localrepair_var5_base", "disflooding_var5_dec",
        "blackhole_var5_base", "disflooding_var5_oo", "localrepair_var15_dec", "disflooding_var10_base", "blackhole_var5_dec", "localrepair_var10_dec",
        "localrepair_var5_dec"]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    if algorithm == "WCL":  
        # if scenario == "random":
        train_WCL.tdim_random(args, run_wandb, train_domains_loader, test_domains_loader, train_domain_order, device,
                            model, exp_no, num_epochs=args.epochs, learning_rate=args.learning_rate, patience=args.patience)           
        # elif scenario in ["w2b", "b2w", "toggle"]:
        #    train_WCL_w2b_b2w_togg.tdim(args, run_wandb, train_domains_loader, test_domains_loader, device,
        #                                model, exp_no, num_epochs=args.epochs, learning_rate=args.learning_rate, patience=args.patience)
        # else:
        #    raise ValueError(f"Unknown scenario for WCL algorithm: {scenario}")
        
    
    elif algorithm == "SI":
        train_si.tdim_si(args, run_wandb, train_domains_loader, test_domains_loader, train_domain_order,
               device, model, exp_no, num_epochs=args.epochs,
               learning_rate=args.learning_rate, patience=args.patience, si_c=args.si_c, si_xi=args.si_xi)
    elif algorithm == "EWC":
        # Train using EWC
        train_ewc.tdim_ewc_random(args, run_wandb, train_domains_loader, test_domains_loader, train_domain_order, 
                                device, model, exp_no, num_epochs=args.epochs, learning_rate=args.learning_rate,
                                patience=args.patience) 
        
    elif algorithm == "LwF":
            train_lwf.tdim_lwf_random(args, run_wandb, train_domains_loader, test_domains_loader, train_domain_order, device,
                                        model, exp_no, num_epochs=args.epochs, learning_rate=args.learning_rate, patience=args.patience,
                                        alpha=args.alpha, T=args.temperature, warmup_epochs=args.warmup_epochs, enc_lr_scale=args.enc_lr_scale,
                                        weight_decay=args.weight_decay)
        
    elif algorithm == "GR":
            train_genreplay.tdim_gr_random(args, run_wandb, train_domains_loader, test_domains_loader, train_domain_order, device,
                                           model, exp_no, num_epochs=args.epochs, learning_rate=args.learning_rate, patience=args.patience,
                                           vae_hidden=args.vae_hidden, vae_latent=args.vae_latent, window_size=args.vae_window_size, num_features=args.num_features,
                                           vae_epochs=args.vae_epochs, vae_lr=args.vae_lr, replay_samples_per_epoch=args.replay_samples_per_epoch,
                                           replay_ratio=args.gr_replay_ratio, use_teacher_labels=args.use_teacher_labels, T=args.distill_T)
            
    elif algorithm == "Replay":
            tdim_replay.tdim_replay( args, run_wandb, train_domains_loader, test_domains_loader, device,model, exp_no, num_epochs=args.epochs,
                                    learning_rate=args.learning_rate, patience=args.patience, replay_total_capacity=args.memory_size, 
                                    replay_per_domain_cap=250, replay_batch_size=128, replay_ratio=0.5, replay_seen_only=True)
            
    
    run_wandb.finish()

if __name__ == "__main__":

    main()
