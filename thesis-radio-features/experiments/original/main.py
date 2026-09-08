import utils
import train
import torch
import os
import logging
import datetime


def main():
    args = utils.parse_args()

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
    exp_no = args.exp_no

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(current_directory, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"exp{exp_no}_log_{timestamp}.log")

    logging.basicConfig(
        filename=log_filename, filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logging.getLogger('').addHandler(console)

    domains_path = os.path.join(os.path.dirname(current_directory), 'attack_data')
    domains = utils.create_domains(domains_path)

    if args.domain != "all":
        if args.domain not in domains:
            raise ValueError(f"Domain '{args.domain}' not found. Available: {list(domains.keys())}")
        domains = {args.domain: domains[args.domain]}

    feature_cols = utils.EXPERIMENT_FEATURES[exp_no]

    if feature_cols is not None:
        num_features = len(feature_cols)
    else:
        sample_domain_key = next(iter(domains))
        folder_key, files = domains[sample_domain_key]
        sample_file = os.path.join(domains_path, folder_key, files[0])
        sample_df = utils.load_csv(sample_file, feature_cols=None)
        num_features = len([c for c in sample_df.columns if c != "label"])

    input_size = num_features * args.window_size

    logging.info(f"Experiment {exp_no} | num_features={num_features} | input_size={input_size}")

    train.train_all_domains(args, domains, domains_path, input_size, feature_cols, device, current_directory, timestamp)


if __name__ == "__main__":
    main()