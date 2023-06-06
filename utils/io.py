import argparse

import yaml


def parse_input():
    """ Parse input arguments """
    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml', help='config file')

    # Network parameters
    parser.add_argument('--hidden_dim', help='net hidden dimensionality', type=int, default=512, required=False)
    parser.add_argument('--num_layers', help='net number of layers', type=int, default=1, required=False)
    parser.add_argument('--latent_size', help='net number of layers', type=int, default=64, required=False)
    parser.add_argument('--latent_combine', help='type of latent vector combination: reduce, concatenate, net',
                        type=str, default='none', required=False)
    parser.add_argument('--lods', help='Number of LoDs', type=int, default=6, required=False)
    parser.add_argument('--device', help='Device: cuda, cpu', type=str, default='cuda', required=False)
    parser.add_argument('--path_net', help='Path to the network if available', type=str, required=False)
    parser.add_argument('--decoder_layers', help='net number of decoder layers', type=int, default=2, required=False)
    parser.add_argument('--curriculum', help='Curriculum training', type=int, default=3, required=False)

    # Optimizer
    parser.add_argument('--learning_rate', help='Learning rate', type=float, default=0.00005, required=False)
    parser.add_argument('--learning_rate_latent', help='Learning rate for the latent embedding',
                        type=float, default=0.005, required=False)
    parser.add_argument('--epochs_max', help='Number of epochs', type=int, default=10000, required=False)
    parser.add_argument('--conf_thres', help='Confidence threshold to jump to next LoD',
                        type=float, default=0.95, required=False)
    parser.add_argument('--scheduler_decay', help='Learning rate multiplicative factor',
                        type=float, default=0.9, required=False)
    parser.add_argument('--scheduler_step', help='Period of learning rate decay',
                        type=float, default=1000, required=False)

    # Loss weights
    parser.add_argument('--w_occ', help='Occupancy loss weight', type=float, default=1, required=False)
    parser.add_argument('--w_sdf', help='SDF loss weight', type=float, default=1, required=False)
    parser.add_argument('--w_nrm', help='NRM loss weight', type=float, default=0.1, required=False)

    # Data
    parser.add_argument('--path_data', help='Path to training data', type=str, required=False)
    parser.add_argument('--dataset_type', help='Dataset type: OctDB', type=str, default='OctDB', required=False)
    parser.add_argument('--cpu_threads', help='CPU threads for the dataloader', type=int, default=0, required=False)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=32, required=False)

    # Evaluation
    parser.add_argument('--iter_log', help='Analyze iter', type=int, default=10, required=False)
    parser.add_argument('--epoch_analyze', help='Analyze epoch', type=int, default=1000, required=False)
    parser.add_argument('--path_output', help='Output path', type=str, default='log/demo', required=False)
    parser.add_argument('--visualize', help='Visualize evaluation', type=bool, default=False, required=False)
    parser.add_argument('--wandb', help='Wandb project name', type=str, required=False)

    # Read default, config and inline arguments
    args = vars(parser.parse_args())
    args_default = {k: parser.get_default(k) for k in args}
    args_inline = {k: v for (k, v) in args.items() if v != args_default[k]}
    args_config = yaml.load(open(args['config']), Loader=yaml.FullLoader)

    # Update default arguments with config and inline arguments (inline has a priority)
    args = args_default.copy()
    args.update(args_config)
    args.update(args_inline)
    args = argparse.Namespace(**args)

    return args
