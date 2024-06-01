# Copyright (c) Meta Platforms, Inc. and affiliates

import os

import jax
import jax.numpy as jnp
import numpy as np
import torch as th


import dataclasses
from typing import Iterator

import ott

from abc import ABC, abstractmethod

from manifold_ot.geometries import Sphere
import gdown

import scanpy as sc

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

def get_bounds(name):
    # This could be cleaned up and better-merged with the bounds in geometries.
    if name == 'scarvelis_circle':
        bounds = (-1.5, 1.5)
        xbounds = ybounds = bounds
    elif name == 'scarvelis_vee':
        xbounds = (-2.5, 15)
        ybounds = (-15, 15)
        bounds = (jnp.array((xbounds[0], ybounds[0])),
                  jnp.array((xbounds[1], ybounds[1])))
    elif name == 'scarvelis_xpath':
        bounds = (-1.5, 1.5)
        xbounds = ybounds = bounds
    elif name == 'snow_goose':
        xbounds = (0, 12)
        ybounds = (0, 12)
        bounds = (jnp.array((xbounds[0], ybounds[0])),
                  jnp.array((xbounds[1], ybounds[1])))
    elif name == 'scrna_2d':
        xbounds = (-15, 16)
        ybounds = (-10, 15)
        bounds = (jnp.array((xbounds[0], ybounds[0])),
                  jnp.array((xbounds[1], ybounds[1])))
    elif name == 'gsb_gmm':
        xbounds = ybounds = bounds = (-20, 20)
    else:
        raise ValueError(f'Invalid data choice: {name}')

    return bounds, xbounds, ybounds

def get_samplers(geometry_str, batch_size, key):
    if 'lsb' in geometry_str or 'babymaze' in geometry_str:
        #return get_lsb_line_finite_samplers(key, batch_size)
        return get_lsb_line_samplers(key, batch_size)
    elif geometry_str in ['gsb_gmm', 'gmm'] or geometry_str == 'neural_net_metric':
        k1, k2, key = jax.random.split(key, 3)
        scale = 16.
        variance = 1.0
        source_sampler = iter(GaussianMixture(
            name='simple', batch_size=batch_size, init_rng=k1, scale=scale, variance=variance))
        target_sampler = iter(GaussianMixture(
            name='circle', batch_size=batch_size, init_rng=k1, scale=scale, variance=variance))
        return source_sampler, target_sampler
    else:
        if 'sq_euclidean' in geometry_str:
            variance = 0.5
            source_mean = jnp.array([-1.0, 0.0])
            target_mean = jnp.array([0.0, 1.0])
        elif geometry_str == 'scarvelis_circle':
            variance = 0.3
            source_mean = jnp.array([-1.0, 0.0])
            target_mean = jnp.array([1.0, 0.0])
        elif geometry_str == 'gsb_vneck':
            variance = 0.2
            source_mean = jnp.array([-7.0, 0.0])
            target_mean = jnp.array([7.0, 0.0])
        elif geometry_str == 'gsb_stunnel':
            variance = 0.5
            source_mean = jnp.array([-11.0, -1.0])
            target_mean = jnp.array([11.0, -1.0])
        else:
            raise ValueError(f'Invalid geometry choice: {geometry_str}')

        k1, k2, key = jax.random.split(key, 3)
        source_sampler = iter(Gaussian(
            batch_size=batch_size, init_key=k1, mean=source_mean, variance=variance))
        target_sampler = iter(Gaussian(
            batch_size=batch_size, init_key=k2, mean=target_mean, variance=variance))
        return source_sampler, target_sampler

    raise ValueError(f'Invalid geometry choice: {geometry_str}')


def get_gsb_gmm_sampler(batch_size, key):
    source_sampler = GaussianMixture(name='simple', batch_size=batch_size, init_rng=jax.random.PRNGKey(0))
    return train_dataloaders


def get_samplers_scarvelis(geometry_str):
    paths = {
        'scarvelis_circle': 'data_gic_24_gaussians_radius_1_std_0p1_100_samples_closed.pt',
        'scarvelis_vee': 'data_mass_split_std_1_100_samples_8_intermediate_scale_x10.pt',
        'scarvelis_xpath': 'data_xpath_std_0p1_100_samples_8_intermediate.pt',
        'snow_goose': 'ebird_snow_goose_data_basemap.npy',
        'scrna_2d': 'schiebinger.npz',
    }
    if geometry_str not in paths:
        raise ValueError(f'Invalid geometry choice: {geometry_str}')

    if 'snow_goose' in geometry_str:
        fname = SCRIPT_PATH+'/../scarvelis_data/'+paths[geometry_str]
        dataset = np.load(fname)
        dataset = jnp.asarray(dataset[0:6])
        
        samplers = [iter(sampler_from_data(dataset[t])) for t in range(dataset.shape[0])]
        return samplers
    elif 'scrna_2d' in geometry_str:
        url = "https://drive.google.com/file/d/1VC9i5gvZAxCE-RkydXHdanXohY6OGO5P/view?usp=sharing"
        output = "../scarvelis_data/schiebinger.npz"
        fname = SCRIPT_PATH+'/../scarvelis_data/'+paths[geometry_str]
        if not os.path.exists(fname):
            gdown.download(url, output, quiet=False, fuzzy=True)
        ### pre-processing as in Scarvelis and Solomon (2023)
        datadict = np.load(fname)
        data_2d = (datadict['original_embedding'])*1e-3
        timestamps = datadict["sample_labels"]
        times = jnp.unique(timestamps)
        subsampled_data_list = []
        data_list = []
        # Generate training data by randomly selecting 500 observations per time point
        for t in list(times):
            data_t = data_2d[timestamps == t]
            data_list.append(data_t)
            np.random.shuffle(data_t) # shuffle rows of data_t
            subsampled_data_t = data_t[:500] # keep only first 500 samples
            subsampled_data_list.append(subsampled_data_t)
        subsampled_data = jnp.asarray(subsampled_data_list)
        samplers = [iter(sampler_from_data(subsampled_data[t])) for t in range(subsampled_data.shape[0])]
        return samplers
    else:
        fname = SCRIPT_PATH+'/../scarvelis_data/'+paths[geometry_str]
        dataset = th.load(fname, map_location='cpu').detach()
        dataset = jnp.asarray(dataset)
        if geometry_str == 'scarvelis_xpath':
            assert dataset.shape[0] == 2
            dataset = jnp.concatenate((dataset[0], dataset[1]), axis=1)

        samplers = [iter(sampler_from_data(dataset[t])) for t in range(dataset.shape[0])]
        return samplers 

def get_samplers_dna(geometry_str,leave_out,batchsize,dim=5,whiten=True):
    paths = {
        'embrio': 'ebdata_v3.h5ad',
        'cite': 'op_cite_inputs_0.h5ad',
        'multi': 'op_train_multi_targets_0.h5ad',
    }
    if geometry_str not in paths:
        raise ValueError(f'Invalid geometry choice: {geometry_str}')

    if geometry_str == 'embrio':
        adata = sc.read_h5ad(SCRIPT_PATH+'/../single_cell_data/datasets/ebdata_v3.h5ad')
        adata.obs["day"] = adata.obs["sample_labels"].cat.codes
    elif geometry_str == 'cite':
        adata = sc.read_h5ad(SCRIPT_PATH+'/../single_cell_data/datasets/op_cite_inputs_0.h5ad')
    elif geometry_str == 'multi':
        adata = sc.read_h5ad(SCRIPT_PATH+'/../single_cell_data/datasets/op_train_multi_targets_0.h5ad')
    else:
        NotImplementedError(f'name: {geometry_str} is not implemented')
    times = adata.obs["day"].unique()
    coords = adata.obsm["X_pca"][:,:dim]
    if whiten:
        mu = coords.mean(axis=0, keepdims=True)
        sigma = coords.std(axis=0, keepdims=True)
        coords = (coords - mu) / sigma
        inv_scaler = lambda _x: _x
    else:
        mu = coords.mean(axis=0, keepdims=True)
        sigma = np.max(coords.std(axis=0, keepdims=True))
        coords = (coords - mu) / sigma
        inv_scaler = lambda _x: _x*sigma + mu
    adata.obsm["X_pca_standardized"] = coords
    X = [
        adata.obsm["X_pca_standardized"][adata.obs["day"] == t]
        for t in times
    ]
    ####leave_out \in \{1,2,3\} out of [0,1,2,3,4]
    Xtrain = []
    Xval = []
    for k in range(leave_out+1):
        if k == leave_out:
            Xtrain.append(X[k+1])
            sampler_test = iter(sampler_from_data(X[k], batch_size=batchsize))
        else:
            Xtrain.append(X[k])

    times = np.linspace(0.0,1.0,len(Xtrain)).tolist()
    samplers_train = [iter(sampler_from_data(Xtrain[t],batch_size=batchsize)) for t in range(len(Xtrain))]

    return samplers_train, sampler_test, inv_scaler, times


@dataclasses.dataclass
class Gaussian:
    batch_size: int
    init_key: jax.random.PRNGKey
    mean: jnp.ndarray
    variance: float = 0.5

    def __iter__(self) -> Iterator[jnp.array]:
        """Random sample generator from Gaussian mixture.
        Returns:
        A generator of samples from the Gaussian mixture.
        """
        return self._create_sample_generators()

    def _create_sample_generators(self) -> Iterator[jnp.array]:
        key = self.init_key
        while True:
            key1, key = jax.random.split(key, 2)
            normal_samples = jax.random.normal(key1, [self.batch_size, 2])
            samples = self.mean  + self.variance ** 2 * normal_samples
            yield samples

@dataclasses.dataclass
class GaussianFiniteSample:
    batch_size: int
    init_key: jax.random.PRNGKey
    mean: jnp.ndarray
    variance: float = 0.5

    def __iter__(self) -> Iterator[jnp.array]:
        """Random sample generator from Gaussian mixture.
        Returns:
        A generator of samples from the Gaussian mixture.
        """
        return self._create_sample_generators()

    def _create_sample_generators(self) -> Iterator[jnp.array]:
        key = self.init_key
        normal_samples = jax.random.normal(key, [self.batch_size, 2])
        while True:
            samples = self.mean  + self.variance ** 2 * normal_samples
            yield samples

def get_lsb_line_samplers(key, batch_size):
    def source_generator(key):
        while True:
            k1, k2, key = jax.random.split(key, 3)
            x1 = jax.random.uniform(k1, (batch_size, 1), minval=-1.25, maxval=-1.)
            x2 = jax.random.uniform(k2, (batch_size, 1), minval=-1., maxval=1.)
            x = jnp.concatenate([x1, x2], axis=1)
            yield x

    def target_generator(key):
        while True:
            k1, k2, key = jax.random.split(key, 3)
            x1 = jax.random.uniform(k1, (batch_size, 1), minval=1, maxval=1.25)
            x2 = jax.random.uniform(k2, (batch_size, 1), minval=-1., maxval=1.)
            x = jnp.concatenate([x1, x2], axis=1)
            yield x

    k1, k2, key = jax.random.split(key, 3)
    source_sampler = iter(source_generator(k1))
    target_sampler = iter(target_generator(k2))
    return source_sampler, target_sampler


def get_lsb_line_finite_samplers(key,batch_size):
    def source_generator(key1):
        while True:
            x1 = jax.random.uniform(key1, (batch_size, 1), minval=-1.25, maxval=-1.)
            x2 = jax.random.uniform(key1, (batch_size, 1), minval=-1., maxval=1.)
            x = jnp.concatenate([x1, x2], axis=1)
            yield x

    def target_generator(key2):
        while True:
            x1 = jax.random.uniform(key2, (batch_size, 1), minval=1, maxval=1.25)
            x2 = jax.random.uniform(key2, (batch_size, 1), minval=-1., maxval=1.)
            x = jnp.concatenate([x1, x2], axis=1)
            yield x

    k1, k2, key = jax.random.split(key, 3)
    source_sampler = iter(source_generator(k1))
    target_sampler = iter(target_generator(k2))
    return source_sampler, target_sampler  

@dataclasses.dataclass
class GaussianMixture:
  """A mixture of Gaussians.

  Args:
    name: the name specifying the centers of the mixture components:

      - ``simple`` - data clustered in one center,
      - ``circle`` - two-dimensional Gaussians arranged on a circle,
      - ``square_five`` - two-dimensional Gaussians on a square with
        one Gaussian in the center, and
      - ``square_four`` - two-dimensional Gaussians in the corners of a
        rectangle

    batch_size: batch size of the samples
    init_rng: initial PRNG key
    scale: scale of the individual Gaussian samples
    variance: the variance of the individual Gaussian samples
  """
  name: str
  batch_size: int
  init_rng: jax.random.PRNGKeyArray
  scale: float = 5.0
  variance: float = 0.5

  def __post_init__(self):
    gaussian_centers = {
        "simple":
            np.array([[0, 0]]),
        "circle":
            np.array([
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
                (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
                (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
                (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            ]),
        "square_five":
            np.array([[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]]),
        "square_four":
            np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),
    }
    if self.name not in gaussian_centers:
      raise ValueError(
          f"{self.name} is not a valid dataset for GaussianMixture"
      )
    self.centers = gaussian_centers[self.name]

  def __iter__(self) -> Iterator[jnp.array]:
    """Random sample generator from Gaussian mixture.

    Returns:
      A generator of samples from the Gaussian mixture.
    """
    return self._create_sample_generators()

  def _create_sample_generators(self) -> Iterator[jnp.array]:
    rng = self.init_rng
    while True:
      rng1, rng2, rng = jax.random.split(rng, 3)
      means = jax.random.choice(rng1, self.centers, [self.batch_size])
      normal_samples = jax.random.normal(rng2, [self.batch_size, 2])
      samples = self.scale * means + self.variance ** 2 * normal_samples
      yield samples


def sampler_from_data(data, batch_size=None):
    while True:
        if batch_size is None:
            yield data
        else:
            idx = np.random.choice(data.shape[0], batch_size, replace=False)
            yield data[idx]
            #assert False # sample from data


# class SphereSampler(ABC):
#     manifold: Sphere
#     @abstractmethod
#     def sample(self, key, n_samples):
#         pass
#     def __hash__(self): return 0 # For jitting

@dataclasses.dataclass
class SphereUniform(ABC):
    manifold: Sphere
    batch_size: int
    init_rng: jax.random.PRNGKeyArray
    def __iter__(self) -> Iterator[jnp.array]:
        return self._create_sample_generators()
    
    def _create_sample_generators(self) -> Iterator[jnp.array]:
        rng = self.init_rng
        while True:
            rng1, rng2, rng = jax.random.split(rng,3)
            xs = jax.random.normal(rng1, shape=[self.batch_size, self.manifold.D])
            samples = self.manifold.project(xs)
            yield samples

@dataclasses.dataclass
class WrappedNormal(ABC):
    manifold: Sphere
    batch_size: int
    init_rng: jax.random.PRNGKeyArray
    loc: jnp.ndarray
    scale: jnp.ndarray

    def __iter__(self) -> Iterator[jnp.array]:
        return self._create_sample_generators()
    
    def _create_sample_generators(self) -> Iterator[jnp.array]:
        rng = self.init_rng
        while True:
            rng1, rng2, rng = jax.random.split(rng,3)
            v = self.scale * jax.random.normal(rng1, [self.batch_size, self.manifold.D-1])
            v = self.manifold.unsqueeze_tangent(v)
            x = self.manifold.zero_like(self.loc)
            u = self.manifold.transp(x, self.loc, v)
            z = self.manifold.exponential_map(self.loc, u)
            yield z

    def __hash__(self): return 0 # For jitting
