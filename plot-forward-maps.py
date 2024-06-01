#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates

import cloudpickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def plot(exps, plot_fname):
    nrow, ncol = 1, len(exps)
    fig, axs = plt.subplots(
        nrow, ncol, figsize=(ncol*4, nrow*4),
        gridspec_kw={'wspace': 0.0, 'hspace': 0.0},
    )
    if len(exps) == 1:
        axs = [axs]

    for ((name, path), ax) in zip(exps, axs):
        print(name)
        exp_fname = os.path.join(path, 'best.pkl')
        if not os.path.exists(exp_fname):
            exp_fname = os.path.join(path, 'latest.pkl')
        if not os.path.exists(exp_fname):
            raise ValueError(f'Could not find {exp_fname}')

        with open(exp_fname, 'rb') as f:
            W = pkl.load(f)

        if W.geometry.xbounds is not None:
            xlims = W.geometry.xbounds
            ylims = W.geometry.ybounds
        else:
            xlims = ylims = W.geometry.bounds
        W.geometry.add_plot_background(
            W.params_geometry, ax, xlims=xlims, ylims=ylims)


        W.neural_dual_solver.plot_forward_map(
            W.source_samples_eval,
            W.target_samples_eval,
            W.state_source_map,
            W.state_target_potential,
            W.params_geometry,
            ax=ax,
            legend=False,
        )
        if len(name) > 0:
            ax.set_title(name)
        else:
            ax.set_title('')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)


    print(f'Saving to {plot_fname}')
    fig.savefig(
        plot_fname, dpi=300,
        bbox_inches='tight', pad_inches=0.0,
        transparent=True,
    )

fname = 'plots/nlsb.pdf'
exps = [
    ('box', 'exp/2023.05.14/2007/13'),
    ('slit', 'exp/2023.05.14/2007/14'),
    ('hill', 'exp/2023.05.14/2007/15'),
    ('well', 'exp/2023.05.14/2007/10'),
]
plot(exps, fname)

for name, path in exps:
    fname = f'plots/{name}.pdf'
    plot([(name, path)], fname)

fname = 'plots/gsb.pdf'
exps = [
    ('', 'exp/2023.05.12/1437/5'),
]
plot(exps, fname)


fname = 'plots/circle1.pdf'
exps = [
    ('', 'exp/2023.05.14/2007/0'),
]
plot(exps, fname)

fname = 'plots/circle2.pdf'
exps = [
    ('', 'exp/2023.05.14/2007/12'),
]
plot(exps, fname)
