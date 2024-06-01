#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates

import argparse

import matplotlib.pyplot as plt
import numpy as np

import pickle as pkl
import os

parser = argparse.ArgumentParser()
parser.add_argument('exp_roots', type=str, nargs='+')
args = parser.parse_args()

nrow, ncol = 1, len(args.exp_roots)
fig, axs = plt.subplots(nrow, ncol, figsize=(ncol*4, nrow*4),
                        gridspec_kw={'wspace': 0, 'hspace': 0})

axs = np.atleast_1d(axs).ravel()

for i, exp_root in enumerate(args.exp_roots):
    fname = os.path.join(exp_root, 'latest.pkl')
    with open(fname, 'rb') as f:
        W = pkl.load(f)

    ax = axs[i]
    W._setup_ax(ax)
    W._clean_axis(ax)

    N = len(W.eval_samples)
    cmap = plt.get_cmap('Blues')
    for i, eval_samples in enumerate(W.eval_samples):
        cmap_idx = 1-i/N
        cmap_idx = cmap_idx/0.9 + 0.1
        color = cmap(cmap_idx)
        ax.scatter(eval_samples[:, 0], eval_samples[:, 1], s=1, color=color)
    # import ipdb; ipdb.set_trace()

fname = 'plots/learned-metrics.pdf'
print(f'saving to {fname}')
fig.savefig(fname, bbox_inches='tight', pad_inches=0)
plt.close(fig)

nrow, ncol = 1, len(args.exp_roots)
fig, axs = plt.subplots(nrow, ncol, figsize=(ncol*4, nrow*4),
                        gridspec_kw={'wspace': 0, 'hspace': 0})

axs = np.atleast_1d(axs).ravel()

for i, exp_root in enumerate(args.exp_roots):
    fname = os.path.join(exp_root, 'latest.pkl')
    with open(fname, 'rb') as f:
        W = pkl.load(f)

    ax = axs[i]
    W.plot_pushforward(ax=ax, fname=None)

fname = 'plots/pushforwards.pdf'
print(f'saving to {fname}')
fig.savefig(fname, bbox_inches='tight', pad_inches=0)
plt.close(fig)
