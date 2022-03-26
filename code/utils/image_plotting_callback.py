import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib.pyplot import figure, imshow
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from torchvision.utils import make_grid


class ImageSampler(pl.Callback):
    def __init__(self):
        super().__init__()
        self.img_size = None
        self.num_preds = 16

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        figure(figsize=(8, 3), dpi=300)

        # Z COMES FROM NORMAL(0, 1)
        z = torch.randn(self.num_preds, pl_module.hparams.latent_dim)

        # SAMPLE IMAGES
        with torch.no_grad():
            pred = pl_module.resnet_decoder(z.to(pl_module.device)).cpu()

        # UNDO DATA NORMALIZATION
        normalize = cifar10_normalization()
        mean, std = np.array(normalize.mean), np.array(normalize.std)
        img = make_grid(pred).permute(1, 2, 0).numpy() * std + mean

        # PLOT IMAGES
        trainer.logger.experiment.add_image(
            "img", torch.tensor(img).permute(2, 0, 1), global_step=trainer.global_step
        )
