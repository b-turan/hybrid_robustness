#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


def save_lr_fig(trainer, model):
    """
    This function replaces pytorchs method "lr_finder.plot()".
    Due to the SSH connection we need to save the plot instead of showing it.

    trainer (torch): initialized pytorch lightning trainer
    model (torch): pytorch model

    """
    lr_finder = trainer.tuner.lr_find(
        model, early_stop_threshold=None, min_lr=1e-06, max_lr=1e-3, mode="exponential"
    )
    lrs = lr_finder.results["lr"]
    losses = lr_finder.results["loss"]
    fig, ax = plt.subplots()
    # Plot loss as a function of the learning rate
    ax.plot(lrs, losses, label="beta = 1e1")
    if lr_finder.mode == "exponential":
        ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Loss")
    if True:
        _ = lr_finder.suggestion()
        if lr_finder._optimal_idx:
            ax.plot(
                lrs[lr_finder._optimal_idx],
                losses[lr_finder._optimal_idx],
                markersize=10,
                marker="o",
                color="red",
            )
    ax.grid(True, alpha=0.3)
    ax.set_title("Optimal lr for orig. architecture on cifar10 (beta=1e5)")
    fig.savefig("lr_plot.png")
