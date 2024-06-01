#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates

import argparse
import jax
import jax.numpy as jnp
from jax import nn
import numpy as np
import os
import optax
import cloudpickle as pkl
from copy import copy
import flax
import json
import time
import shutil
import csv
import torch
import functools

from geomloss import SamplesLoss

import dataclasses
from typing import Iterator

from ott.geometry import pointcloud
from ott.solvers.linear import sinkhorn

import hydra

from manifold_ot import models, neuraldual, metrics, geodesics, geometries, data, spline_amortizer

import matplotlib.pyplot as plt
plt.style.use('bmh')

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Neutral', call_pdb=1)


class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")

        if self.cfg.save_all_plots:
            os.makedirs('plots')

        self.train_step = 0
        self.key = jax.random.PRNGKey(self.cfg.seed)

        self.geometry = geometries.get(
            self.cfg.geometry, self.cfg.geometry_kwargs)

        if self.cfg.data is None:
            self.cfg.data = self.cfg.geometry
        source_sampler, target_sampler = data.get_samplers(
            self.cfg.data, self.cfg.batch_size, self.key)
        if isinstance(self.geometry, geometries.SqEuclidean):
            self.geometry.bounds, self.geometry.xbounds, self.geometry.ybounds = data.get_bounds(self.cfg.data)

        num_eval_samples = 1024 if self.cfg.data in ['gsb_gmm', 'gmm'] else 128
        source_sampler_eval, target_sampler_eval = data.get_samplers(
            self.cfg.data, num_eval_samples, self.key)
        self.source_samples_eval = next(source_sampler_eval)
        self.target_samples_eval = next(target_sampler_eval)

        k1, self.key = jax.random.split(self.key)
        self.params_geometry = self.geometry.init(
            k1, self.source_samples_eval[0], self.target_samples_eval[0],
            method=self.geometry.cost
        ).get('params', {})
        self.geometry_has_annealing = 'lagrangian_potential_module' in self.params_geometry
        self.anneal_params_geometry()
        if 'spline_model' in self.params_geometry:
            self.fit_spline_amortizer(
                source_sampler=source_sampler,
                target_sampler=target_sampler,
                init=True)

        self.target_potential = models.MLP(
            is_potential=True, **self.cfg.target_potential_kwargs,
        )
        self.source_map = models.MLP(
            is_potential=False, **self.cfg.source_map_kwargs,
        )
        self.ctransform_solver = hydra.utils.instantiate(self.cfg.ctransform_solver)

        self.neural_dual_solver = neuraldual.ManifoldW2NeuralDual(
            geometry=self.geometry,
            target_potential=self.target_potential,
            source_map=self.source_map,
            ctransform_solver=self.ctransform_solver,
        )

        self.optimizer_target_potential = get_opt(self.cfg.num_train_iters, self.cfg.target_potential_opt)
        self.optimizer_source_map = get_opt(self.cfg.num_train_iters, self.cfg.source_map_opt)


        init_key, self.key = jax.random.split(self.key)
        self.state_target_potential, self.state_source_map = self.neural_dual_solver.initialize_states(
            self.optimizer_target_potential,
            self.optimizer_source_map,
            init_key, self.source_samples_eval, self.target_samples_eval)

        self.elapsed_time = 0.
        self.best_marginal_w2 = np.inf

    def fit_spline_amortizer(self, source_sampler, target_sampler, init):
        num_iters = self.cfg.spline.init_train_iters if init else self.cfg.spline.train_iters

        if init:
            def sampler(key):
                # sample from random pairs of source and target
                while True:
                    source_samples = next(source_sampler)
                    target_samples = next(target_sampler)
                    all_samples = jnp.concatenate([source_samples, target_samples], axis=0)
                    k1, key = jax.random.split(key)
                    all_samples = jax.random.permutation(k1, all_samples)
                    yield all_samples

            k1, k2, self.key = jax.random.split(self.key, 3)
            xsampler = iter(sampler(k1))
            ysampler = iter(sampler(k2))
        else:
            xsampler = source_sampler
            def ysampler():
                key = jax.random.PRNGKey(0)
                while True:
                    source_samples = next(source_sampler)
                    transported_samples = self.neural_dual_solver.source_map_apply_jit(
                        {'params': self.state_source_map.params}, source_samples)
                    # seems too expensive to call into the ctransform solver every time:
                    # transported_samples = self.neural_dual_solver.pushforward_jit_vmap(
                    #     self.state_source_map.params,
                    #     self.state_target_potential.params,
                    #     source_samples).solution
                    if self.cfg.spline.noise > 0.:
                        k1, key = jax.random.split(key)
                        transported_samples += self.cfg.spline.noise * jax.random.normal(
                            key, transported_samples.shape)
                    yield transported_samples
            ysampler = iter(ysampler())

        self.params_geometry = self.geometry.spline_amortizer.train(
            self.params_geometry,
            xsampler, ysampler,
            max_iter=num_iters,
            grad_norm_threshold=self.cfg.spline.grad_norm_threshold,
        )


    def anneal_params_geometry(self):
        current_t = self.train_step
        max_t = self.cfg.anneal_geometry_steps

        if self.cfg.anneal_geometry_steps is None \
                or self.cfg.anneal_geometry_steps <= 0 \
                or not self.geometry_has_annealing \
                or max_t < current_t:
            return

        t = current_t / max_t
        assert 0. <= t and t <= 1.

        self.params_geometry = self.params_geometry.unfreeze()
        self.params_geometry['lagrangian_potential_module'] = \
            self.geometry.lagrangian_potential_initializer_fn.get_annealed_params(t)
        self.params_geometry = flax.core.freeze(self.params_geometry)
        print('updated lagrangian params: ',
              ', '.join([f'{k}: {v.item():.2e}' for k,v in \
                        self.params_geometry['lagrangian_potential_module'].items()]))


    def run(self):
        source_sampler, target_sampler = data.get_samplers(
            self.cfg.data, self.cfg.batch_size, self.key)
        self.plot()

        logf, writer = self._init_logging()

        while self.train_step < self.cfg.num_train_iters:
            start_time = time.time()
            train_batch = {
                "source": jnp.asarray(next(source_sampler)),
                "target": jnp.asarray(next(target_sampler)),
            }
            self.anneal_params_geometry()


            new_states, info = self.neural_dual_solver.update_fn_jit(
                self.state_target_potential,
                self.state_source_map,
                self.params_geometry,
                train_batch,
            )
            self.state_target_potential, self.state_source_map = new_states
            update_step_time = time.time() - start_time
            self.elapsed_time += update_step_time
            print(
                f'step: {self.train_step}/{self.cfg.num_train_iters} '
                f'dual_loss: {info.dual_loss:.2e}, amor_loss: {info.amor_loss:.2e} '
                f'num_ctransform_iter: {info.num_ctransform_iter:.2f} '
                f'update_step_time: {update_step_time:.2f}s '
            )

            start_time = time.time()
            if self.cfg.spline.restart_frequency is not None \
                    and self.train_step % self.cfg.spline.restart_frequency == 0 \
                    and self.train_step < self.cfg.num_train_iters - 1000:
                print('restarting spline amortizer')
                new_params_geometry = self.params_geometry.unfreeze()
                k1, self.key = jax.random.split(self.key)
                new_params_geometry['spline_model'] = self.geometry.init(
                    k1, self.source_samples_eval[0], self.target_samples_eval[0],
                    method=self.geometry.cost
                )['params']['spline_model']
                self.params_geometry = flax.core.freeze(new_params_geometry)

            if self.cfg.spline.update_on_conjugates \
                    and 'spline_model' in self.params_geometry:
                self.params_geometry = self.geometry.spline_amortizer.train_single(
                    self.params_geometry,
                    train_batch['source'], info.target_hat,
                )

            if self.train_step % self.cfg.spline.update_frequency == 0 \
                    and 'spline_model' in self.params_geometry \
                    and self.train_step < self.cfg.num_train_iters - 1000:
                self.fit_spline_amortizer(
                    source_sampler=source_sampler,
                    target_sampler=target_sampler, init=False)

            if self.train_step % self.cfg.plot_frequency == 0:
                self.plot()

                if self.cfg.save_all_plots:
                    shutil.copy('latest.png', f'plots/{self.train_step:06d}.png')
                    if self.train_step % (10*self.cfg.save_frequency) == 0 \
                            and self.train_step > 1000:
                        os.system('ffmpeg -r 10 -pattern_type glob -i "plots/*.png" -c:v '
                                'libx264 -pix_fmt yuv420p -crf 23 -y training.mp4')

            if self.train_step % self.cfg.eval_frequency == 0:
                marginal_w2 = self.eval_marginal_W2()

                writer.writerow({
                    'iter': self.train_step,
                    'ot_cost': -info.dual_loss,
                    'marginal_w2': marginal_w2,
                    'elapsed_time': self.elapsed_time,
                })
                logf.flush()

                if marginal_w2 < self.best_marginal_w2 and \
                        (not self.geometry_has_annealing or \
                         self.train_step > self.cfg.anneal_geometry_steps):
                    self.best_marginal_w2 = marginal_w2
                    print(f'updating best marginal_w2: {self.best_marginal_w2:.2e}')
                    shutil.copy('latest.png', 'best.png')
                    self.save('best')


            self.elapsed_time += time.time() - start_time

            # update the step and save last
            self.train_step += 1
            if self.train_step % self.cfg.save_frequency == 0:
                self.save()


    def eval_marginal_W2(self):
        print('evaluating W2 of pushforward to target')
        transported_samples = self.neural_dual_solver.pushforward_jit_vmap(
            self.state_source_map.params,
            self.state_target_potential.params,
            self.params_geometry,
            self.source_samples_eval).solution
        ot_cost = self.sinkhorn_cost(self.target_samples_eval, transported_samples).item()
        print(f'distance: {ot_cost:.2e}')
        return ot_cost

    def plot(self):
        print('plotting')

        nrow, ncol = 1, 2
        fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 4, nrow * 4))
        if self.geometry.xbounds is not None:
            xlims = self.geometry.xbounds
            ylims = self.geometry.ybounds
        else:
            xlims = ylims = self.geometry.bounds
        for ax in axs:
            self.geometry.add_plot_background(
                self.params_geometry, ax, xlims=xlims, ylims=ylims)
            ax.set_aspect('equal')
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)

        max_plot_samples = 1024
        self.neural_dual_solver.plot_forward_map(
            self.source_samples_eval[:max_plot_samples],
            self.target_samples_eval[:max_plot_samples],
            self.state_source_map,
            self.state_target_potential,
            self.params_geometry,
            ax=axs[0],
        )

        self.neural_dual_solver.plot_target_potential(
            self.source_samples_eval[:max_plot_samples],
            self.target_samples_eval[:max_plot_samples],
            self.state_target_potential,
            ax=axs[1],
            x_bounds=xlims,
            y_bounds=ylims,
        )

        fig.savefig('latest.png')
        plt.close(fig)


    def save(self, tag="latest"):
        path = os.path.join(self.work_dir, f"{tag}.pkl")
        print(f"Saving to {path}")
        with open(path, "wb") as f:
            pkl.dump(self, f)


    def _init_logging(self):
        logf = open('log.csv', 'a')
        fieldnames = ['iter', 'ot_cost', 'marginal_w2', 'elapsed_time']
        writer = csv.DictWriter(logf, fieldnames=fieldnames)
        if os.stat('log.csv').st_size == 0:
            writer.writeheader()
            logf.flush()
        return logf, writer

    @functools.partial(jax.jit, static_argnums=(0,))
    def sinkhorn_cost_ott(self, x, y):
        geom = pointcloud.PointCloud(x, y, epsilon=1e-3)
        ot = sinkhorn.solve(geom, a=None, b=None)
        return ot.reg_ot_cost

    def sinkhorn_cost_geomloss(self, x, y):
        sinkhorn = SamplesLoss("sinkhorn", p=2, blur=5e-3, scaling=0.9)
        x = torch.from_numpy(np.asarray(x))
        y = torch.from_numpy(np.asarray(y))
        return sinkhorn(x, y)

    def sinkhorn_cost(self, x, y):
        if 'gmm' in self.cfg.data:
            return self.sinkhorn_cost_geomloss(x, y)
        else:
            return self.sinkhorn_cost_ott(x, y)



def get_opt(num_train_iters, cfg):
    if cfg.alpha is not None and cfg.alpha > 0.:
        lr = optax.cosine_decay_schedule(
            init_value=cfg.init_lr,
            decay_steps=num_train_iters,
            alpha=cfg.alpha,
        )
    else:
        lr = cfg.init_lr
    opt = optax.adamw(
        learning_rate=lr,
        b1=cfg.b1, b2=cfg.b2)
    if cfg.grad_clip is not None:
        opt = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            opt
        )
    return opt


@hydra.main(config_path=".", config_name="train_ot.yaml", version_base="1.1")
def main(cfg):
    from train_ot import Workspace as W

    fname = os.getcwd() + "/latest.pkl"
    if os.path.exists(fname):
        print(f"Resuming fom {fname}")
        with open(fname, "rb") as f:
            workspace = pkl.load(f)
    else:
        workspace = W(cfg)

    workspace.run()

if __name__ == '__main__':
    main()
