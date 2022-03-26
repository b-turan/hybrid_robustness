#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytorch_lightning as pl
import sklearn.metrics as metrics
import torch
import torchvision
from pl_bolts.models.autoencoders import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms

from utils import arg_parser
from utils.ece_metric import _ECELoss
from utils.image_plotting_callback import ImageSampler
from utils.lr_finder_custom import save_lr_fig

parser = arg_parser.create_parser()
args = parser.parse_args()

if args.seed == True:
    pl.seed_everything(42)


class ZModel(pl.LightningModule):
    """
    Hybrid Model, where a classifier (MLP) is applied to the latent variable z of a VAE. 
    Additional Features: 
    - Training Datasets: SVHN, CIFAR10
    - Accuracy with respect to FGSM attacks
    - OOD Datasets: SVHN, CIFAR100
    - OOD Detection via max p(y|x) as in https://arxiv.org/abs/1610.02136 (using ARUOC)
    - OOD Detection via log p(x) (using AUROC)
    - Beta-VAE, see https://openreview.net/forum?id=Sy2fzU9gl
    """

    def __init__(
        self,
        loss_coeffs,
        enc_out_dim: int = 512,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
        lr: float = 1e-3,
        num_class: int = 10,
        eps_interval=[0.02, 0.08],
    ):
        super(ZModel, self).__init__()
        # self.save_hyperparameters()
        self.loss_coeffs = loss_coeffs  # alpha, beta
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.num_class = num_class
        self.DATA_PATH = args.data
        self.accuracy = pl.metrics.Accuracy()
        self.ece_criterion = _ECELoss(eval=args.eval)

        self.eps_interval = eps_interval
        # ResNet-Backbones
        self.resnet_decoder = resnet18_decoder(self.latent_dim, 32, first_conv=False, maxpool1=False)
        self.resnet_encoder = resnet18_encoder(first_conv=False, maxpool1=False)  # resnet18-backbone
        # classifier head
        self.final = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.latent_dim, int(self.latent_dim / 2)),
            nn.Dropout(0.25),
            nn.Linear(int(self.latent_dim / 2), num_class),
        )
        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)
        # CIFAR-100 Testset
        if args.ood_cifar100:
            self.ood_dataloader100 = self.get_ood_data("cifar100")
            self.ood_dataiter100 = iter(self.ood_dataloader100)
        # SVHN Testset
        if args.ood_svhn:
            self.ood_dataloader_svhn = self.get_ood_data("svhn")
            self.ood_dataiter_svhn = iter(self.ood_dataloader_svhn)

    def forward(self, x):
        x_vae = self.resnet_encoder(x)
        mu = self.fc_mu(x_vae)
        logits = self.final(mu)
        return logits

    def _run_step(self, x):
        """
        Returns z, x_hat, p, q, logits, eps, std. 
        """
        x_vae = self.resnet_encoder(x)
        mu = self.fc_mu(x_vae)
        log_var = self.fc_var(x_vae)
        p, q, z, eps, std = self.sample(mu, log_var)  # with importance sampling
        logits = self.final(mu)
        # vanilla vae
        x_hat = self.resnet_decoder(z)
        return z, x_hat, p, q, logits, eps, std, mu

    def sample(self, mu, log_var):
        """
        Returns distributios q and p and samples z. For IWAE use importance_num > 1.

        mu (torch.tensor): mean of z
        log_var (torch.tensor): log of variance of z

        """
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        eps = Variable(std.data.new(args.importance_num, std.size()[0], std.size()[1]).normal_())
        eps = eps.view(std.size()[0], std.size()[1])  # adjust dimensions
        z = mu + std * eps
        return p, q, z, eps, std

    def vae_loss_fun(self, x, x_hat, z, p, q, eps, std, mu):
        """
        Calculates the loss function for the VAE.
        Additionally the sample wise ELBO values are returned.

        Returns total VAE loss, reconstruction loss, ELBO (sample wise)
        """
        batch_size = x.shape[0]
        # vanilla beta-vae loss calculation
        recon_lossx = torch.zeros(batch_size).cuda()  # initialization
        for i in range(batch_size):
            recon_lossx[i] = F.mse_loss(x_hat[i], x[i], reduction="mean")  # sample wise recon loss
        kl = torch.distributions.kl_divergence(q, p)
        klx = kl.mean(1)  # sample wise
        klx *= self.kl_coeff  # beta-vae
        loss_vaex = klx + recon_lossx  # sample wise
        elbo = -loss_vaex  # sample-wise elbo for vanilla beta-vae
        loss_vae = loss_vaex.mean()  # total vae loss
        recon_loss = recon_lossx.mean()
        return loss_vae, recon_loss, elbo

    def step(self, batch, batch_idx, validation):
        """
        Train/Validation step.
        """
        x, y = batch
        alpha, beta = self.loss_coeffs
        z, x_hat, p, q, logits, eps, std, mu = self._run_step(x)
        loss_vae, recon_loss, elbo = self.vae_loss_fun(x, x_hat, z, p, q, eps, std, mu)
        # loss of classification
        criterion = nn.CrossEntropyLoss()
        loss_classification = criterion(logits, y)
        if beta > 0:
            # loss of vae
            loss = alpha * loss_classification + beta * loss_vae
        elif beta == 0:
            loss = loss_classification

        # accuracy of predictions
        y_hat = F.softmax(logits, dim=1)
        softmax_x, y_hat = torch.max(y_hat, dim=1)
        acc = self.accuracy(y_hat, y)
        # expected calibration error
        ece = self.ece_criterion(logits, y, args.eval)
        logs = {
            "recon_loss": recon_loss,
            "loss_vae": loss_vae,
            "loss_classification": loss_classification,
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
                    self.ood_batch, _ = self.ood_dataiter100.next()
                except StopIteration:
                    # if not possible, reinitialize dataloader
                    self.ood_dataiter100 = iter(self.ood_dataloader100)
                    self.ood_batch, _ = self.ood_dataiter100.next()

                self.ood_batch = self.ood_batch.cuda()
                ood_score_PyGx_cifar100 = self.eval_ood_PyGx(softmax_x, ood_batch=self.ood_batch)
                logs.update({"ood_score_PyGx_cifar100": ood_score_PyGx_cifar100})
                if beta > 0:
                    ood_score_logPx_cifar100 = self.eval_ood_logPx(elbo, ood_batch=self.ood_batch)
                    logs.update({"ood_score_logPx_cifar100": ood_score_logPx_cifar100})
            if args.ood_svhn:
                # load next ood-batches or reinitialize dataloaders
                try:
                    # try next batch
                    self.ood_batch, _ = self.ood_dataiter_svhn.next()
                except StopIteration:
                    # if not possible, reinitialize dataloader
                    self.ood_dataiter_svhn = iter(self.ood_dataloader_svhn)
                    self.ood_batch, _ = self.ood_dataiter_svhn.next()

                self.ood_batch = self.ood_batch.cuda()
                ood_score_logPyGx_svhn = self.eval_ood_PyGx(softmax_x, ood_batch=self.ood_batch)
                logs.update({"ood_score_PyGx_svhn": ood_score_logPyGx_svhn})
                if beta > 0:
                    ood_score_logPx_svhn = self.eval_ood_logPx(elbo, y, y_hat, ood_batch=self.ood_batch)
                    logs.update({"ood_score_logPx_svhn": ood_score_logPx_svhn})
        return loss, logs

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

    def eval_ood_PyGx(self, softmax_x, ood_batch):
        """
        Evaluate OOD detection via max p(y|x) as in Hendrycks 2016 (SVHN vs. CIFAR100/CIFAR10)
        Use In-Distribution samples as positive class (e.g. "1") and OOD-Data as negative class (e.g. "0") for OOD-Detection.
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

    def eval_ood_logPx(self, elbo, ood_batch):
        """
        Evaluate OOD detection via log(p(x)) using the ELBO of the generative model.
        Use In-Distribution samples as positive class (e.g. "1") and OOD-Data as negative class (e.g. "0") for OOD-Detection.
        Calculate Area Under the Receiver Operating Characteristic Curve (AU-ROC).
        ------
        batch (val_loader batch): batch of in distribution validation set
        batch_idx (val loader batch_idx): batch index of in distribtion validation set
        ------
        Returns AU-ROC score.
        """
        with torch.no_grad():
            # in distribution
            y_positive = torch.ones(len(elbo)).cuda()  # positive class
            # out of distribution
            x_ood = ood_batch[: len(elbo)]
            z_ood, x_hat_ood, p_ood, q_ood, _, eps_ood, std_ood, mu = self._run_step(x_ood)
            _, _, elbo_ood = self.vae_loss_fun(x_ood, x_hat_ood, z_ood, p_ood, q_ood, eps_ood, std_ood, mu)
            y_negative = torch.zeros(len(elbo_ood)).cuda()  # negative class
            # calculate Area Under the Receiver Operating Characteristic curve (AUROC)
            y_true = torch.cat((y_positive, y_negative))
            y_scores = torch.cat((elbo, elbo_ood))
            roc_auc_score = metrics.roc_auc_score(y_true.cpu(), y_scores.cpu())
            roc_auc_score = torch.tensor(roc_auc_score).cuda()
        return roc_auc_score

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
        optimizer = torch.optim.Adam(self.parameters(), lr=(self.lr))
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
            self.train_set, batch_size=args.batch_size, shuffle=True, num_workers=40, pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.validation_set, batch_size=args.batch_size, num_workers=40, pin_memory=True,
        )
        return val_loader

    def test_dataloader(self):
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),]
        )
        # test dataset / SVHN
        test_set = datasets.SVHN(args.data, split="test", download=True, transform=transform_test)
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
                testset_cifar100, batch_size=args.batch_size, shuffle=False, num_workers=40
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
                testset_svhn, batch_size=args.batch_size, shuffle=False, num_workers=40
            )
        return ood_dataloader


def z_model(loss_coeffs: list, lr: float, eps_interval: list):
    """
    Returns supervised hdgm model. 

    loss_coeffs (list): [alpha, beta]
    lr (float): learning rate
    """
    model = ZModel(loss_coeffs=loss_coeffs, lr=lr, eps_interval=eps_interval)  # initialize model
    return model


def run():
    # trainer settings
    # trainer settings
    if args.logger == "csv_logger":
        logger = pl_loggers.CSVLogger(save_dir=args.output_path, name="csv_logs")
    elif args.logger == "tb_logger":
        logger = pl_loggers.TensorBoardLogger(save_dir=args.output_path, default_hp_metric=False)
    checkpoint_callback = ModelCheckpoint(monitor=args.monitor, mode="max")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # sampler = ImageSampler()
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        logger=logger,
        callbacks=[checkpoint_callback],  # optional: sampler
        fast_dev_run=args.fast_dev_run,
    )

    if args.eval == True:
        # testing
        PATH = "../lightning_logs/supervised_hdgm/results/svhn_200epochs_lr_1e-4_multiple_runs/baseline/396991/default/version_0/checkpoints/epoch=191-step=28223.ckpt"
        model = ZModel.load_from_checkpoint(PATH)
        trainer.test(model, verbose=True)
    else:
        # training
        model = z_model(args.loss_coeffs, args.lr, eps_interval=args.eps_interval)
        trainer.fit(model)


if __name__ == "__main__":
    run()
