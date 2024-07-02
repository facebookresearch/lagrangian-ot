#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates

import os
import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Neutral', call_pdb=1)

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
# from jax.config import config
# config.update("jax_enable_x64", True)

import numpy as np

import time

from dataclasses import dataclass

import shutil
import glob

import functools

import matplotlib.pyplot as plt
from lagrangian_ot import geometries, metrics, splines, lagrangian_potentials

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PLOT_DIR = SCRIPT_DIR + '/plots/geodesics'

def sampler_from_data(data):
    while True:
        yield data

def plot_euclidean():
    print("--- Euclidean")
    m = geometries.MetricManifold(
        metric_initializer_fn=metrics.EuclideanMetric)
    params_geometry = m.init(
        jax.random.PRNGKey(0),
        jnp.zeros(2), jnp.zeros(2),
        method=m.cost
    )['params']

    bounds = (-2., 2.)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')

    y = jnp.array([1., 0.])

    @jax.jit
    @jax.vmap
    def geodesic_to_y(x):
        return m.apply({'params': params_geometry}, x, y, method=m.path)

    n = 100
    mean = jnp.array([-1., 0.])
    xs = mean + 0.3*jax.random.normal(jax.random.PRNGKey(0), (n, 2))
    ys = jnp.tile(y, (n, 1))

    xsampler = iter(sampler_from_data(xs))
    ysampler = iter(sampler_from_data(ys))
    params_geometry = m.spline_amortizer.train(
        params_geometry,
        xsampler, ysampler,
        max_iter=5001,
        grad_norm_threshold=1e-5,
    )

    geodesics = geodesic_to_y(xs)
    # print(f'average number of iterations: {jnp.mean(out.num_iter): .2f}')

    ax.scatter(xs[:,0], xs[:,1], color='k')
    ax.scatter(y[0], y[1], color='k')

    for geo_xs in geodesics:
        ax.plot(geo_xs[:,0], geo_xs[:,1], color='k', alpha=0.5)


    fname = PLOT_DIR+'/euclidean.png'
    print(f'saving to {fname}')
    fig.tight_layout()
    fig.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)


def plot_circle():
    print("--- Circle")
    m = geometries.MetricManifold(
        metric_initializer_fn=metrics.CircleMetric)
    params_geometry = m.init(
        jax.random.PRNGKey(0),
        jnp.zeros(2), jnp.zeros(2),
        method=m.cost
    )['params']

    bounds = (-2., 2.)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')

    m.add_plot_background(params_geometry, ax, xlims=bounds)

    y = jnp.array([1., 0.])

    @jax.jit
    @jax.vmap
    def geodesic_to_y(x):
        return m.apply({'params': params_geometry}, x, y, method=m.path)

    n = 100
    mean = jnp.array([-1., 0.])
    xs = mean + 0.3*jax.random.normal(jax.random.PRNGKey(0), (n, 2))
    ys = jnp.tile(y, (n, 1))

    xsampler = iter(sampler_from_data(xs))
    ysampler = iter(sampler_from_data(ys))
    params_geometry = m.spline_amortizer.train(
        params_geometry,
        xsampler, ysampler,
        max_iter=5001,
        grad_norm_threshold=1e-5,
    )

    geodesics = geodesic_to_y(xs)
    # print(f'average number of iterations: {jnp.mean(out.num_iter):.2f}')

    ax.scatter(xs[:,0], xs[:,1], color='k')
    ax.scatter(y[0], y[1], color='k')

    for geo_xs in geodesics:
        ax.plot(geo_xs[:,0], geo_xs[:,1], color='k', alpha=0.5)

    fname = PLOT_DIR+'/circle.png'
    print(f'saving to {fname}')
    fig.tight_layout()
    fig.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)


def plot_vee():
    print("--- Vee")
    m = geometries.MetricManifold(metric_initializer_fn=metrics.VeeMetric)
    params_geometry = m.init(
        jax.random.PRNGKey(0),
        jnp.zeros(2), jnp.zeros(2),
        method=m.cost
    )['params']

    xbounds = (-3, 16)
    ybounds = (-15, 15)
    fig, ax = plt.subplots(figsize=(4, 4))

    m.add_plot_background(params_geometry, ax, xlims=xbounds, ylims=ybounds)

    y = jnp.array([0., 0.])

    @jax.jit
    @jax.vmap
    def geodesic_to_y(x):
        return m.apply({'params': params_geometry}, x, y, method=m.path)

    n = 100
    xs = np.zeros((n, 2))
    xs[:,0] = 10. + 0.3*np.random.randn(n)
    xs[:,1] = np.linspace(-10, 10, n)
    ys = jnp.tile(y, (n, 1))

    xsampler = iter(sampler_from_data(xs))
    ysampler = iter(sampler_from_data(ys))
    params_geometry = m.spline_amortizer.train(
        params_geometry,
        xsampler, ysampler,
        max_iter=5001,
        grad_norm_threshold=1e-5,
    )

    geodesics = geodesic_to_y(xs)
    # print(f'average number of iterations: {jnp.mean(out.num_iter):.2f}')

    ax.scatter(xs[:,0], xs[:,1], color='k')
    ax.scatter(y[0], y[1], color='k')

    for geo_xs in geodesics:
        ax.plot(geo_xs[:,0], geo_xs[:,1], color='k', alpha=0.5)

    fname = PLOT_DIR+'/vee.png'
    print(f'saving to {fname}')
    fig.tight_layout()
    fig.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)


def plot_x():
    print("--- X")
    m = geometries.MetricManifold(metric_initializer_fn=metrics.XMetric)
    params_geometry = m.init(
        jax.random.PRNGKey(0),
        jnp.zeros(2), jnp.zeros(2),
        method=m.cost
    )['params']

    bounds = (-2, 2)
    fig, ax = plt.subplots(figsize=(4, 4))

    m.add_plot_background(params_geometry, ax, xlims=bounds)

    y = jnp.array([-1., -1.])

    @jax.jit
    @jax.vmap
    def geodesic_to_y(x):
        return m.apply({'params': params_geometry}, x, y, method=m.path)

    n = 100
    xs = np.zeros((n, 2))
    xs[:,0] = np.linspace(-1.8, 1.8, n)
    xs[:,1] = 1.5 + 0.1*np.random.randn(n)
    ys = jnp.tile(y, (n, 1))

    xsampler = iter(sampler_from_data(xs))
    ysampler = iter(sampler_from_data(ys))
    params_geometry = m.spline_amortizer.train(
        params_geometry,
        xsampler, ysampler,
        max_iter=5001,
        grad_norm_threshold=1e-5,
    )

    geodesics = geodesic_to_y(xs)
    # print(f'average number of iterations: {jnp.mean(out.num_iter):.2f}')

    ax.scatter(xs[:,0], xs[:,1], color='k')
    ax.scatter(y[0], y[1], color='k')

    for geo_xs in geodesics:
        ax.plot(geo_xs[:,0], geo_xs[:,1], color='k', alpha=0.5)

    fname = PLOT_DIR+'/x.png'
    print(f'saving to {fname}')
    fig.tight_layout()
    fig.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)


def plot_potential(name, potential):
    print(f"--- {name} potential")
    m = geometries.MetricManifold(
        metric_initializer_fn=metrics.EuclideanMetric,
        lagrangian_potential_initializer_fn=potential,
        distance_mode=geometries.DistanceModes.LAGRANGIAN
    )
    params_geometry = m.init(
        jax.random.PRNGKey(0),
        jnp.zeros(2), jnp.zeros(2),
        method=m.cost
    )['params']

    if name in ['hill', 'well']:
        bounds = (-2., 2.)
    else:
        bounds = (-1.5, 1.5)

    fig, ax = plt.subplots(figsize=(4, 4))

    m.add_plot_background(params_geometry, ax, xlims=bounds)

    n = 100
    xs = np.zeros((n, 2))
    xs[:,0] = np.random.uniform(-1.2, -1., n)
    xs[:,1] = np.linspace(-1, 1, n)

    ys = np.zeros((n, 2))
    ys[:,0] = np.random.uniform(1., 1.2, n)
    ys[:,1] = np.linspace(-1, 1, n)


    xsampler = iter(sampler_from_data(xs))
    ysampler = iter(sampler_from_data(ys))

    params_geometry = m.spline_amortizer.train(
        params_geometry,
        xsampler, ysampler,
        max_iter=5001,
        grad_norm_threshold=1e-5,
    )

    geodesics = jax.jit(jax.vmap(
        functools.partial(
            m.apply, {'params': params_geometry}, method=m.path
    )))(xs, ys)
    # print(f'average number of iterations: {jnp.mean(geodesics.num_iter):.2f}')

    for i in range(n):
        geo_xs = geodesics[i]
        ax.scatter(xs[i,0], xs[i,1], color='k')
        ax.scatter(ys[i,0], ys[i,1], color='k')
        ax.plot(geo_xs[:,0], geo_xs[:,1], color='k', alpha=0.5)

    fname = PLOT_DIR+f'/{name}.png'
    print(f'saving to {fname}')
    fig.tight_layout()
    fig.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)


def main():
    # shutil.rmtree(PLOT_DIR, ignore_errors=True)
    # os.makedirs(PLOT_DIR, exist_ok=True)

    # plot_euclidean()
    # plot_circle()
    # plot_vee()
    # plot_x()

    all_potentials = {
        # 'box': lagrangian_potentials.BoxPotential,
        # 'slit': lagrangian_potentials.SlitPotential,
        # 'well': lagrangian_potentials.WellPotential,
        'hill': lagrangian_potentials.HillPotential,
        # 'babymaze': lagrangian_potentials.BabyMazePotential,
    }
    for name, potential in all_potentials.items():
        plot_potential(name, potential)

    montage_fname = PLOT_DIR+'/all.png'
    if os.path.exists(montage_fname):
        os.remove(montage_fname)

    plots = glob.glob(PLOT_DIR+'/*.png')
    os.system('montage -tile 4x -geometry 400x400+0+0 '+' '.join(plots)+' '+montage_fname)
    print(f'saving to {montage_fname}')


if __name__ == '__main__':
    main()
