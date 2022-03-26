#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import numpy as np
import pytorch_lightning as pl
import sklearn.metrics as metrics
import torch
import torchvision
from pl_bolts.models.autoencoders import resnet18_encoder
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms

from utils import arg_parser
from utils.ece_metric import _ECELoss

parser = arg_parser.create_parser()
args = parser.parse_args()


class ResNet(pl.LightningModule):
    """
    ResNet Classifier with fully connected head
    """

    def __init__(
        self,
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        latent_dim: int = 256,
        lr: float = 1e-3,
        num_class: int = 10,
        eps_interval=[0.02, 0.08],
    ):
        super(ResNet, self).__init__()
        # self.save_hyperparameters()

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.num_class = num_class
        self.DATA_PATH = args.data
        self.eps_interval = eps_interval
        self.ece_criterion = _ECELoss(eval=args.eval)


        self.accuracy = pl.metrics.Accuracy()

        self.resnet_classifier = resnet18_encoder(first_conv, maxpool1)
        self.final = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.enc_out_dim, self.latent_dim),
            nn.Dropout(0.25),
            nn.Linear(self.latent_dim, num_class),
        )

        # CIFAR-100 Testset
        if args.ood_cifar100:
            self.ood_dataloader100 = self.get_ood_data("cifar100")
            self.ood_dataiter100 = iter(self.ood_dataloader100)
        # SVHN Testset
        if args.ood_svhn:
            self.ood_dataloader_svhn = self.get_ood_data("svhn")
            self.ood_dataiter_svhn = iter(self.ood_dataloader_svhn)

    def forward(self, x):
        x = self.resnet_classifier(x)
        logits = self.final(x)
        return logits

    def step(self, batch, batch_idx, validation):
        x, y = batch
        logits = self(x)

        # loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)

        # accuracy of predictions
        y_hat = F.softmax(logits, dim=1)
        softmax_x, y_hat = torch.max(y_hat, dim=1)
        acc = self.accuracy(y_hat, y)

        # expected calibration error
        ece = self.ece_criterion(logits, y, args.eval)

        logs = {
            "loss": loss,
            "accuracy": acc,
            "ece": ece,
        }

        if validation == True:
            if args.fgsm:
                # adversarial accuracy
                logs = self.fast_gradient_sign_method(
                    logs, x, criterion, y, epsilon_interval=self.eps_interval
                )
            # load next ood-batches or reinitialize dataloaders
            if args.ood_cifar100:
                try:
                    # try next batch
                    self.images100, _ = self.ood_dataiter100.next()
                except StopIteration:
                    # if not possible, reinitialize dataloader
                    self.ood_dataiter100 = iter(self.ood_dataloader100)
                    self.images100, _ = self.ood_dataiter100.next()

                self.images100 = self.images100.cuda()
                roc_auc_score_cifar100 = self.eval_ood_PyGx(softmax_x, ood_batch=self.images100)
                # add ood-score to logs
                logs.update({"ood_score_PyGx_cifar100": roc_auc_score_cifar100})

            if args.ood_svhn:
                try:
                    # try next batch
                    self.images_svhn, _ = self.ood_dataiter_svhn.next()
                except StopIteration:
                    # if not possible, reinitialize dataloader
                    self.ood_dataiter_svhn = iter(self.ood_dataloader_svhn)
                    self.images_svhn, _ = self.ood_dataiter_svhn.next()

                self.images_svhn = self.images_svhn.cuda()
                roc_auc_score_svhn = self.eval_ood_PyGx(softmax_x, ood_batch=self.images_svhn)
                # add ood-score to logs
                logs.update({"roc_auc_score_svhn": roc_auc_score_svhn})

            if args.eval == True:
                    self.acc_bin = self.ece_criterion.acc_bin
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx, validation=False)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx, validation=True)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def test_step(self, batch, batch_idx):
        _, logs = self.step(batch, batch_idx, validation=True)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 50))
        return [optimizer], [scheduler]

    def prepare_data(self):
        if args.dataset == "cifar10":
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
            # training dataset
            self.train_set = datasets.CIFAR10(
                self.DATA_PATH, train=True, download=True, transform=transform_train
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ]
            )
            # validation dataset
            self.validation_set = datasets.CIFAR10(self.DATA_PATH, train=False, transform=transform_test)

        elif args.dataset == "svhn":
            # TODO normalization of SVHN dataset
            transform_train = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),]
            )
            svhn = datasets.SVHN(self.DATA_PATH, split="train", download=True, transform=transform_train)
            # randomly split into 80/20
            self.train_set, self.validation_set = torch.utils.data.random_split(svhn, [58605, 14652])

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=400, shuffle=True, num_workers=40, pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.validation_set, batch_size=400, num_workers=40, pin_memory=True,
        )
        return val_loader

    def test_dataloader(self):
        if args.dataset == "cifar10":
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ]
            )
            test_set = datasets.CIFAR10(
                self.DATA_PATH, train=False, transform=transform_test
            )  # 10k datapoints

        elif args.dataset == "svhn":
            transform_test = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),]
            )
            # test dataset / SVHN
            test_set = datasets.SVHN(self.DATA_PATH, split="test", download=True, transform=transform_test)

        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, num_workers=40, pin_memory=True,
        )
        return test_loader

    def get_ood_data(self, ood_data):
        """
        Returns dataloader for out-of-distribution detection.
        ------
        ood_data (str): choice of out of distribution data ("cifar100" or "svhn")
        """
        if ood_data == "cifar100":
            # download CIFAR-100 Testset
            transform_cifar100 = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
                ]
            )
            testset_cifar100 = torchvision.datasets.CIFAR100(
                root=(self.DATA_PATH), train=False, download=True, transform=transform_cifar100
            )
            ood_dataloader = torch.utils.data.DataLoader(
                testset_cifar100, batch_size=400, shuffle=False, num_workers=40
            )
        elif ood_data == "svhn":
            # download svhn Testset
            transform_svhn = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),]
            )
            testset_svhn = torchvision.datasets.SVHN(
                root=(self.DATA_PATH), split="test", download=True, transform=transform_svhn
            )

            ood_dataloader = torch.utils.data.DataLoader(
                testset_svhn, batch_size=400, shuffle=False, num_workers=40
            )
        return ood_dataloader

    def eval_ood_PyGx(self, softmax_x, ood_batch):
        """
        Evaluate OOD detection via max p(y|x) as in Hendrycks 2016 (SVHN vs. CIFAR100)
        Seperates In-Distribution Data in true positive and false negative classified images.
        Use correctly classified images as positive class (e.g. "1") and OOD-Data as negative class (e.g. "0") for OOD-Detection.
        Utilizes softmaxes of predicted classes to calculate Area Under the Receiver Operating Characteristic Curve (AU-ROC).
        ------
        softmax_x (torch.tensor): predicted softmax values
        ood_batch (torch.tensor): Out of distribution batch; svhn/cifar100
        ------
        Returns AU-ROC score.
        """

        with torch.no_grad():
            # in distribution (no mask)
            y_positive = torch.ones(len(softmax_x)).cuda()  # positive class
            # out of distribution
            x_ood = ood_batch[: len(softmax_x)]
            ood_outputs = self(x_ood)
            ood_probabilites = F.softmax(ood_outputs, dim=1)
            ood_softmaxes, _ = torch.max(ood_probabilites, dim=1)
            y_negative = torch.zeros(len(ood_softmaxes)).cuda()  # negative class
            # calculate Area Under the Receiver Operating Characteristic curve (AUROC)
            y_true = torch.cat((y_positive, y_negative))
            y_scores = torch.cat((softmax_x, ood_softmaxes))
            roc_auc_score = metrics.roc_auc_score(y_true.cpu(), y_scores.cpu())
            roc_auc_score = torch.tensor(roc_auc_score).cuda()
        return roc_auc_score

    def fast_gradient_sign_method(self, logs, x, criterion, y, epsilon_interval=[0.02, 0.08]):
        with torch.enable_grad():
            inp_img = x.clone().requires_grad_()
            logits = self(inp_img)
            self.zero_grad()
            loss_class = criterion(logits, y)
            loss_class.backward()
            # Update image to adversarial example
            noise_grad = torch.sign(inp_img.grad)
            inp_img.grad = None
            n = int(100 * (epsilon_interval[1] - epsilon_interval[0]) + 1)
            for epsilon in np.linspace(epsilon_interval[0], epsilon_interval[1], n):
                adv_imgs = inp_img + epsilon * noise_grad
                adv_imgs.detach_()
                adv_logits = self(adv_imgs)
                adv_y_hat = F.softmax(adv_logits, dim=1)
                _, adv_y_hat = torch.max(adv_y_hat, dim=1)
                adv_acc = self.accuracy(adv_y_hat, y)
                adv_acc_key = "adv_accuracy_{}".format(epsilon)
                logs.update({adv_acc_key: adv_acc})
        return logs


def parser():
    # argparser
    parser = argparse.ArgumentParser(description="ResNet-Model with fully connected head")
    # Trainer args  (gpus, epochs etc.)
    parser.add_argument("-e", "--epochs", type=int, metavar="", help="Number of Epochs", default=150)
    parser.add_argument(
        "-g", "--gpus", type=int, metavar="", help="Number of GPUS, (None for CPU)", default=1
    )
    parser.add_argument(
        "-fdr", "--fast_dev_run", type=bool, metavar="", help="Fast dev mode", default=False,
    )
    parser.add_argument("--eval", type=bool, metavar="", help="Eval mode on/off", default=False)
    # Model specific arguments
    parser.add_argument("--batch_size", type=int, metavar="", help="Batch Size", default=400)
    parser.add_argument("--seed", type=bool, metavar="", help="Seed everything on/off", default=False)
    parser.add_argument("--lr", type=float, metavar="", help="Learning Rate", default=1e-3)
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
        default="../lightning_logs/resnet_classifier/",
    )
    args = parser.parse_args()
    return args


def run(args):
    # trainer settings
    if args.logger == "csv_logger":
        logger = pl_loggers.CSVLogger(save_dir="../lightning_logs/resnet_classifier/", name="csv_logs")
    elif args.logger == "tb_logger":
        logger = pl_loggers.TensorBoardLogger(
            save_dir="../lightning_logs/resnet_classifier/", default_hp_metric=False
        )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        fast_dev_run=args.fast_dev_run,
    )

    if args.eval == True:
        # testing
        PATH = args.checkpoint_path
        model = ResNet.load_from_checkpoint(PATH)
        trainer.test(model, verbose=True)
    else:
        # training
        model = ResNet(lr=args.lr)
        trainer.fit(model)


if __name__ == "__main__":
    # args = parser()
    # if args.seed == True:
    #     pl.seed_everything(42)
    run(args)
