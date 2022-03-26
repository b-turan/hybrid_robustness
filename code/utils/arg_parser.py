import argparse
from collections import defaultdict


def create_parser():
    parser = argparse.ArgumentParser(description="Hybrid-Model Original Architecture")
    # Trainer args  (gpus, epochs etc.)
    parser.add_argument("-e", "--epochs", type=int, metavar="", help="Number of Epochs", default=150)
    parser.add_argument(
        "-g", "--gpus", type=int, metavar="", help="Number of GPUS, (None for CPU)", default=1
    )
    parser.add_argument(
        "-fdr", "--fast_dev_run", type=bool, metavar="", help="Fast dev mode", default=False,
    )
    parser.add_argument("--monitor", type=str, metavar="", help="Monitoring Metric", default="val_accuracy")
    parser.add_argument("--eval", type=bool, metavar="", help="Eval mode on/off", default=False)
    parser.add_argument(
        "--fgsm", type=bool, metavar="", help="Fast Gradient Sign Method on/off", default=False
    )
    parser.add_argument(
        "--eps_interval",
        nargs=2,
        type=float,
        metavar="",
        help="Epsilon Values for FGSM [bottom, top]",
        default=[0.02, 0.08],
    )
    parser.add_argument(
        "--logger",
        type=str,
        metavar="",
        help="Logger choice",
        default="csv_logger",
        choices=["csv_logger", "tb_logger"],
    )
    # Model specific arguments
    parser.add_argument("--lr", type=float, metavar="", help="Learning Rate", default=1e-3)
    parser.add_argument(
        "--vae_split_layer",
        type=int,
        metavar="",
        choices=[0, 1, 2, 3, 4],
        help="layer at which the split into classifier and vae lies",
        default=0,
    )
    parser.add_argument("--batch_size", type=int, metavar="", help="Batch Size", default=400)
    parser.add_argument("--seed", type=bool, metavar="", help="Seed everything on/off", default=False)
    parser.add_argument(
        "--loss_coeffs",
        nargs=2,
        type=float,
        metavar="",
        help="Loss Coefficients [alpha, beta]",
        default=[1, 1],
    )
    parser.add_argument("--importance_num", type=int, default=1)  # k of IWAE; use k=1 for vanilla VAE
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        metavar="",
        help="Path to checkpoint with highest accuracy",
        default=None,
    )
    # Program arguments (data_path, save_dir, etc.)
    parser.add_argument(
        "--dataset",
        type=str,
        metavar="",
        help="Choice of Dataset for training",
        choices=["svhn", "cifar10"],
        default="svhn",
    )
    parser.add_argument(
        "--ood_svhn", type=bool, metavar="", help="Out of Distribution Dataset SVHN", default=False
    )
    parser.add_argument(
        "--ood_cifar100", type=bool, metavar="", help="Out of Distribution Dataset CIFAR100", default=False,
    )
    parser.add_argument("--data", type=str, metavar="", help="Input Path for Data", default="../data/")
    parser.add_argument(
        "--output_path",
        type=str,
        metavar="",
        help="Directory for saving the Output",
        default="../lightning_logs/supervised_hdgm/",
    )

    parser.add_argument(
        "--histogram", type=bool, metavar="", help="Plot Histogram of p(x) values", default=False,
    )
    parser.add_argument(
        "--histo_csv_path", type=str, metavar="", help="Paths to p(x) data", nargs=2, default=None,
    )
    parser.add_argument("--ece_vals", type=bool, metavar="", help="Save ECE values", default=False)

    parser.add_argument(
        "--acc_bin_csv_path",
        type=str,
        metavar="",
        help="Path to Accuracy of each bin for ECE Plot",
        default=None,
    )
    return parser
