import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pl_bolts.models.autoencoders import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)
from pytorch_lightning import loggers as pl_loggers

from resnet_module import ResNet
from supervised_hdgm import SupervisedHDGM, supervised_hdgm
from utils import arg_parser

parser = arg_parser.create_parser()
args = parser.parse_args()

if args.seed == True:
    pl.seed_everything(42)


def write_csv_histogram(model, trainer):
    # p(x) histogram data
    logPx = model.logPx.cpu()
    logPx_np = logPx.numpy()
    logPx_df = pd.DataFrame(logPx_np)
    # p(x_ood) histogram data
    logPx_ood = model.logPx_ood.cpu()
    logPx_ood_np = logPx_ood.numpy()
    logPx_ood_df = pd.DataFrame(logPx_ood_np)
    # save to file
    logPx_df.to_csv(trainer.logger.log_dir + "/logPx_values.csv")
    logPx_ood_df.to_csv(trainer.logger.log_dir + "/logPx_ood_values.csv")


def write_ece_values(model, trainer):
    acc_bin = model.acc_bin
    acc_bin_np = acc_bin.cpu()
    acc_bin_df = pd.DataFrame(acc_bin_np)
    # save file
    acc_bin_df.to_csv(trainer.logger.log_dir + "/acc_bins.csv")


def run():
    # Trainer settings
    if args.logger == "csv_logger":
        logger = pl_loggers.CSVLogger(save_dir=args.output_path, name="csv_logs")
    elif args.logger == "tb_logger":
        logger = pl_loggers.TensorBoardLogger(save_dir=args.output_path, default_hp_metric=False)
    trainer = pl.Trainer(gpus=args.gpus, logger=logger)
    if args.eval == True:
        # load model parameters
        PATH = args.checkpoint_path  # path to checkpoint
        model = SupervisedHDGM.load_from_checkpoint(PATH) # hparams_file="/data/svhn/checkpoints/hparams.yaml"
        model.eps_interval = args.eps_interval  # define fsgm range
        trainer.test(model, verbose=True)  # test
        if args.histogram == True:
            write_csv_histogram(model, trainer)  # write histogram data to csv-file
        if args.ece_vals == True:
            write_ece_values(model, trainer)
    else:
        print("Evaluation not turned on")


if __name__ == "__main__":
    run()
