#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates

import argparse
import math
import functools

import csv
import time
import json
import jax
import jax.numpy as jnp
import numpy as np
import os
import optax
import cloudpickle as pkl
from copy import copy

import dataclasses
from typing import Iterator

import hydra

from lagrangian_ot import models, neuraldual, metrics, geodesics, geometries, data

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

        self.key = jax.random.PRNGKey(self.cfg.seed)
        self.elapsed_time = 0.

        self.geometry = geometries.get(
            self.cfg.geometry, self.cfg.geometry_kwargs)

        if 'euclidean' in self.cfg.geometry or 'neural' in self.cfg.geometry:
            if self.cfg.data is None:
                raise ValueError(
                    'data must be specified for euclidean and neural geometries')

        self.has_reference_geometry = self.cfg.data in ['scarvelis_circle', 'scarvelis_vee', 'scarvelis_xpath']
        if 'neural' in self.cfg.geometry and self.has_reference_geometry:
            self.reference_geometry = geometries.get(
                self.cfg.data, self.cfg.geometry_kwargs)

        if self.cfg.data is None:
            self.cfg.data = self.cfg.geometry
        samplers = data.get_samplers_scarvelis(self.cfg.data)

        self.geometry.bounds, self.geometry.xbounds, self.geometry.ybounds = data.get_bounds(self.cfg.data)

        self.num_pairs = len(samplers) - 1
        print(f'training on {self.num_pairs} pairs')
        self.eval_samples = [next(s) for s in samplers]

        self.optimizer_target_potential = optax.adamw(
            learning_rate=self.cfg.potential_lr)
        self.optimizer_source_map = self.optimizer_target_potential
        self.optimizer_geom = optax.adamw(
            learning_rate=self.cfg.metric.lr)

        k1, self.key = jax.random.split(self.key)
        self.params_geometry = self.geometry.init(
            k1, self.eval_samples[0][0], self.eval_samples[1][0],
            method=self.geometry.cost
        ).get('params', {})
        self.state_geometry = self.optimizer_geom.init(self.params_geometry)

        target_potential = models.MLP(
            dim_hidden=self.cfg.target_potential_dim_hidden,
            is_potential=True)
        source_map = models.MLP(
            dim_hidden=self.cfg.source_map_dim_hidden,
            is_potential=False)
        ctransform_solver = hydra.utils.instantiate(self.cfg.ctransform_solver)
        self.neural_dual_solver = neuraldual.ManifoldW2NeuralDual(
            geometry=self.geometry,
            target_potential=target_potential,
            source_map=source_map,
            ctransform_solver=ctransform_solver,
        )


        init_key, self.key = jax.random.split(self.key)
        state_target_potential, state_source_map = self.neural_dual_solver.initialize_states(
            self.optimizer_target_potential, self.optimizer_source_map,
            init_key, self.eval_samples[0], self.eval_samples[1])
        self.state_target_potentials = [state_target_potential]
        self.state_source_maps = [state_source_map]

        if 'spline_model' in self.params_geometry:
            self.fit_spline_amortizer(samplers, init=True)

        self.train_step = 0


    def fit_spline_amortizer(self, samplers, init):
        num_iters = self.cfg.spline.init_train_iters if init else self.cfg.spline.train_iters

        if init:
            def sampler(key):
                # sample from random pairs of source and target
                t = 0
                while True:
                    source_samples = next(samplers[t])
                    target_samples = next(samplers[t+1])
                    all_samples = jnp.concatenate([source_samples, target_samples], axis=0)
                    k1, key = jax.random.split(key)
                    all_samples = jax.random.permutation(k1, all_samples)
                    t = (t + 1) % self.num_pairs
                    yield all_samples

            k1, k2, self.key = jax.random.split(self.key, 3)
            xsampler = iter(sampler(k1))
            ysampler = iter(sampler(k2))
        else:
            def xsampler():
                key = jax.random.PRNGKey(0)
                t = 0
                while True:
                    source_samples = next(samplers[t])
                    t = (t + 1) % self.num_pairs
                    yield source_samples

            def ysampler():
                key = jax.random.PRNGKey(0)
                t = 0
                while True:
                    source_samples = next(samplers[t])
                    transported_samples = self.neural_dual_solver.source_map_apply_jit(
                        {'params': self.state_source_maps[t].params}, source_samples)
                    if self.cfg.spline.noise > 0.:
                        k1, key = jax.random.split(key)
                        transported_samples += self.cfg.spline.noise * jax.random.normal(
                            key, transported_samples.shape)
                    t = (t + 1) % self.num_pairs
                    yield transported_samples

            xsampler = iter(xsampler())
            ysampler = iter(ysampler())


        self.params_geometry = self.geometry.spline_amortizer.train(
            self.params_geometry,
            xsampler, ysampler,
            max_iter=num_iters,
            grad_norm_threshold=self.cfg.spline.grad_norm_threshold,
        )



    def update_all_states(self, state_target_potentials, state_source_maps, batches):
        out = []
        for t in range(self.num_pairs):
            out_t = self.neural_dual_solver.update_fn_jit(
                state_target_potentials[t if self.train_step > 0 else 0],
                state_source_maps[t if self.train_step > 0 else 0],
                self.params_geometry,
                batches[t],
            )
            out.append(out_t)

            if self.cfg.spline.update_on_conjugates \
                    and 'spline_model' in self.params_geometry:
                _, info = out_t
                self.params_geometry = self.geometry.spline_amortizer.train_single(
                    self.params_geometry,
                    batches[t]['source'], info.target_hat,
                    verbose=False,
                )

        new_states, infos = zip(*out)
        new_states = zip(*new_states)
        mean_info = type(infos[0])(
            *[jnp.array(x).mean() for x in list(zip(*infos))])
        return new_states, mean_info

    def sample_all_batches(self, samplers):
        batches = []
        for t in range(self.num_pairs):
            batches.append({
                "source": jnp.asarray(next(samplers[t])),
                "target": jnp.asarray(next(samplers[t+1])),
            })
        return batches


    def geometry_loss(self, params_geometry,
                      state_target_potentials, state_source_maps,
                      batches, key):
        metric = lambda x: self.geometry.apply(
            {'params': params_geometry},
            x, method=self.geometry.metric)
        metric_vmap = jax.vmap(metric)
        metric_jac_vmap = jax.vmap(jax.jacfwd(metric))

        dual_losses = []
        for t in range(self.num_pairs):
            batch = batches[t]
            _, info_t = self.neural_dual_solver.loss_fn(
                state_target_potentials[t].params,
                state_source_maps[t].params,
                params_geometry, batch)
            dual_losses.append(-info_t.dual_loss)

        mean_dual_loss = jnp.mean(jnp.stack(dual_losses))
        total_loss = mean_dual_loss
        return total_loss

    @functools.partial(jax.jit, static_argnums=[0])
    def update_geometry(self, params_geometry, state_geometry,
                        state_target_potentials, state_source_maps,
                        batches, key):
        geometry_grad_fn = jax.value_and_grad(self.geometry_loss)
        loss, grads = geometry_grad_fn(
            params_geometry,
            state_target_potentials, state_source_maps,
            batches, key)

        # TODO: could remove 'spline_model' from updates
        # (currently grads are all zero)
        updates, new_state_geometry = self.optimizer_geom.update(
            grads, state_geometry, params=params_geometry)
        new_params_geometry = optax.apply_updates(params_geometry, updates)

        return new_params_geometry, new_state_geometry, loss

    def run(self):
        samplers = data.get_samplers_scarvelis(self.cfg.data)

        logf, writer = self._init_logging()
        dual_loss = -1.

        while self.train_step < self.cfg.num_train_iters:
            start = time.time()
            batches = self.sample_all_batches(samplers)
            new_states, info = self.update_all_states(
                self.state_target_potentials, self.state_source_maps,
                batches)
            self.state_target_potentials, self.state_source_maps = new_states
            update_time = time.time() - start
            self.elapsed_time += update_time
            print(
                f'[{self.train_step}/{self.cfg.num_train_iters}] '
                f'dual_loss: {info.dual_loss:.2e}, amor_loss: {info.amor_loss:.2e} '
                f'num_ctransform_iter: {info.num_ctransform_iter:.2f} '
                f'update_time: {update_time:.2f} '
            )

            start = time.time()
            if self.train_step > self.cfg.metric.warmup_steps \
                    and self.train_step % self.cfg.metric.update_frequency == 0 \
                    and self.cfg.geometry == 'neural_net_metric':
                for _ in range(self.cfg.metric.update_repeat):
                    k1, self.key = jax.random.split(self.key)
                    self.params_geometry, self.state_geometry, dual_loss = self.update_geometry(
                        self.params_geometry, self.state_geometry,
                        self.state_target_potentials, self.state_source_maps,
                        batches, k1)
                    print(f'=== [metric] dual_loss: {dual_loss:.2e}')


            if self.train_step % self.cfg.spline.update_frequency == 0 \
                    and 'spline_model' in self.params_geometry:
                self.fit_spline_amortizer(samplers, init=False)
            self.elapsed_time += time.time() - start

            if self.train_step % self.cfg.plot_frequency == 0:
                self.plot()

                if self.has_reference_geometry:
                    alignment = self.eval_alignment()
                else:
                    alignment = -1.

                writer.writerow({
                    'iter': self.train_step,
                    'potential_dual_loss': info.dual_loss,
                    'metric_dual_loss': dual_loss,
                    'elapsed_time': self.elapsed_time,
                    'alignment': alignment,
                })
                logf.flush()

            self.train_step += 1
            if self.train_step % self.cfg.save_frequency == 0:
                self.save()

        self.plot()
        self.eval_alignment()


    def eval_alignment(self):
        xflat, x1, x2 = geometries._get_grid(
            self.geometry.xbounds, self.geometry.ybounds, 100)
        xflat = jnp.asarray(xflat)

        if not hasattr(self, 'eigvecs'):
            @functools.partial(jax.jit, static_argnums=[0])
            @functools.partial(jax.vmap, in_axes=(None, None, 0))
            def eigvecs(geometry, params_geometry, x):
                A = geometry.apply(
                    {'params': params_geometry},
                    x, method=self.geometry.metric)
                vals, vecs = jnp.linalg.eigh(A)
                return vecs.T
            self.eigvecs = eigvecs

        true_A_evecs = self.eigvecs(
            self.reference_geometry, self.params_geometry, xflat)
        learned_A_evecs = self.eigvecs(
            self.geometry, self.params_geometry, xflat)
        alignment = jnp.abs(
            (true_A_evecs * learned_A_evecs).sum(axis=2)).mean().item()
        return alignment

    def _init_logging(self):
        logf = open('log.csv', 'a')
        fieldnames = ['iter', 'potential_dual_loss', 'metric_dual_loss',
                      'elapsed_time', 'alignment']
        writer = csv.DictWriter(logf, fieldnames=fieldnames)
        if os.stat('log.csv').st_size == 0:
            writer.writeheader()
            logf.flush()
        return logf, writer


    def plot(self):
        print('--- plotting')
        self.plot_all_pairs()
        self.plot_pushforward()

    def plot_all_pairs(self):
        if self.cfg.geometry == 'scarvelis_xpath' or self.cfg.geometry == 'scarvelis_vee':
            ncol = 5
        else:
            ncol = 6
        nrow = math.ceil(self.num_pairs / ncol)
        fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 4, nrow * 4),
                                gridspec_kw={'wspace': 0, 'hspace': 0})
        axs = axs.ravel()
        self._setup_ax(axs)

        for t in range(self.num_pairs):
            self.neural_dual_solver.plot_forward_map(
                self.eval_samples[t],
                self.eval_samples[t+1],
                self.state_source_maps[t],
                self.state_target_potentials[t],
                self.params_geometry,
                ax=axs[t],
                legend=False,
            )

        for ax in axs:
            self._clean_axis(ax)

        fname = 'all-pairs.png'
        print(f'saving to {fname}')
        fig.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def _add_map(self, ax):
        # llcrnrlat, urcrnrlat = 22, 75
        # llcrnrlon, urcrnrlon = -172, -90
        llcrnrlat, urcrnrlat = 22, 90
        llcrnrlon, urcrnrlon = -172, -30

        from mpl_toolkits.basemap import Basemap
        m = Basemap(projection='merc',llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                    llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,
                    suppress_ticks=False, rsphere=5, resolution='l')

        m.drawcoastlines(ax=ax, color='gray')

    def _setup_ax(self, axs):
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        for ax in axs:
            if hasattr(self.geometry, 'xbounds'):
                xlims = self.geometry.xbounds
                ylims = self.geometry.ybounds
            else:
                xlims = ylims = self.geometry.bounds

            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            # ax.set_aspect('equal')

            if self.cfg.data == 'snow_goose':
                self._add_map(ax)

        self.geometry.add_plot_background(
            self.params_geometry, axs, xlims=xlims, ylims=ylims)

        if 'neural' in self.cfg.geometry:
            if self.cfg.data in ['scarvelis_xpath','scarvelis_vee','scarvelis_circle']:
                self.reference_geometry.add_plot_background(
                    self.params_geometry, axs, xlims=xlims, ylims=ylims,
                    alpha=0.5,
                )


    def _clean_axis(self, ax):
        ax.set_title('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color('k')
            sp.set_linewidth(3)

    def plot_pushforward(self, ax=None, fname='pushforward.png', num_samples=100):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))

        self._setup_ax(ax)

        num_samples = 50
        # colors = plt.style.library['bmh']['axes.prop_cycle'].by_key()['color']

        cmap = plt.get_cmap('Blues')
            # cmap_idx = 1-i/N
            # cmap_idx = cmap_idx/0.9 + 0.1
            # color = cmap(cmap_idx)
        # def get_color(i):
        #     import ipdb; ipdb.set_trace()
        colors = cmap(np.linspace(1., 0.1, self.num_pairs+1))

        init_xs = jax.random.choice(
            jax.random.PRNGKey(0), self.eval_samples[0], shape=(num_samples,))
        for i in range(num_samples):
            init_x = init_xs[i]
            ax.scatter([init_x[0]], [init_x[1]], color='k', s=20, alpha=1,
                       zorder=10)

            x = init_x
            for t in range(self.num_pairs):
                prev_x = x
                # x = self.neural_dual_solver.source_map_apply_jit(
                #     {'params': self.state_source_maps[t].params}, x)
                x = self.neural_dual_solver.pushforward_jit(
                    self.state_source_maps[t].params,
                    self.state_target_potentials[t].params,
                    self.params_geometry,
                    x
                ).solution

                path = self.neural_dual_solver.path_jit(
                    self.params_geometry, prev_x, x)

                ax.plot(
                    path[:, 0], path[:, 1],
                    color=colors[t],
                    alpha=0.5,
                    lw=3,
                )

        self._clean_axis(ax)

        if fname is not None:
            fig = ax.get_figure()
            print(f'saving to {fname}')
            fig.savefig(fname, bbox_inches='tight', pad_inches=0)


    def save(self, tag="latest"):
        path = os.path.join(self.work_dir, f"{tag}.pkl")
        print(f"Saving to {path}")
        with open(path, "wb") as f:
            pkl.dump(self, f)


@hydra.main(config_path=".", config_name="train_ot_scarvelis.yaml", version_base="1.1")
def main(cfg):
    from train_ot_scarvelis import Workspace as W

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
