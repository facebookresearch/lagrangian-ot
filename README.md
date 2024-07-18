# [Neural Optimal Transport with Lagrangian Costs](https://arxiv.org/abs/2406.00288)

UAI 2024. [Aram-Alexandre Pooladian](https://arampooladian.com/),
[Carles Domingo-Enrich](https://cdenrich.github.io/),
[Ricky T. Q. Chen](https://rtqichen.github.io/), and
[Brandon Amos](https://bamos.github.io/).

> We investigate the optimal transport problem between probability measures when the underlying cost function is understood to satisfy a least action principle, also known as a Lagrangian cost. These generalizations are useful when connecting observations from a physical system where the transport dynamics are influenced by the geometry of the system, such as obstacles (e.g., incorporating barrier functions in the Lagrangian), and allows practitioners to incorporate a priori knowledge of the underlying system such as non-Euclidean geometries (e.g., paths must be circular). Our contributions are of computational interest, where we demonstrate the ability to efficiently compute geodesics and amortize spline-based paths, which has not been done before, even in low dimensional problems. Unlike prior work, we also output the resulting Lagrangian optimal transport map without requiring an ODE solver. We demonstrate the effectiveness of our formulation on low-dimensional examples taken from prior work.

![output](https://github.com/user-attachments/assets/c6ed8892-5998-47d1-928a-ee97d8e49ae9)

# Setup and dependencies

## `train_ot.py`
The code can be set up with the following commands.
They will install the CPU version of `jax==0.4.13`,
and `jaxlib==0.4.13`, which requires Python ~3.10,
(this jax version doesn't support Python 3.12).
**I recommend manually installing a compatible GPU version
of JAX.** Otherwise the code will run very slow.
For compatibility with the CUDA and cudnn library versions
on my system, I use the GPU version of `jaxlib==0.4.7` in
Python 3.12 with roughly the same versions
of the dependencies in `requirements.txt`.

```bash
conda create -n lagrangian_ot python=3.10
conda activate lagrangian_ot
pip install -r requirements.txt
```

# Reproducing the experiments

## Solving Neural Lagrangian Optimal Transport (NLOT) problems

![image](https://github.com/facebookresearch/lagrangian-ot/assets/707462/67be9cea-8c9c-4a77-9c88-adf7be45a12e)

### Figure 1

![image](https://github.com/facebookresearch/lagrangian-ot/assets/707462/cd8a61aa-f564-405c-bf11-01ddbb7e3650)

```
./train_ot.py geometry=gsb_gmm
./train_ot.py geometry=scarvelis_circle
```

### Figure 2

![image](https://github.com/facebookresearch/lagrangian-ot/assets/707462/dffd562b-c88a-48e3-81a7-5dd1ef5d999f)


```
./train_ot.py geometry=lsb_box
./train_ot.py geometry=lsb_slit
./train_ot.py geometry=lsb_hill
./train_ot.py geometry=lsb_well
```

Multiple seeds can be run using Hydra's multirun mode.
This requires setting a launcher in the Hydra
config `train_ot.yaml`,
which isn't set by default in this repo (I use the `sumitit_slurm` launcher).

```
./train_ot.py -m geometry=lsb_box,lsb_slit,lsb_hill,lsb_well seed=0,1,2
```

## Metric learning with NLOT
![image](https://github.com/facebookresearch/lagrangian-ot/assets/707462/c73b488d-d545-456a-a3bb-159d062343cb)

### Table 2
![image](https://github.com/facebookresearch/lagrangian-ot/assets/707462/4f88d6c8-398d-4ee6-bd94-49e4f4be6819)

```
./train_ot_scarvelis.py geometry=scarvelis_circle
./train_ot_scarvelis.py geometry=scarvelis_vee
./train_ot_scarvelis.py geometry=scarvelis_xpath
```

Multiple seeds can be run using Hydra's multirun mode.
This requires setting a launcher in the Hydra
config `train_ot_scarvelis.yaml`,
which isn't set by default in this repo (I use the `sumitit_slurm` launcher).

```
./train_ot_scarvelis.py -m geometry=scarvelis_circle,scarvelis_vee,scarvelis_xpath seed=0,1,2
```

### Figures 3 and 4
![image](https://github.com/facebookresearch/lagrangian-ot/assets/707462/09cec318-27c2-4b3a-92e2-312251b9cde7)
![image](https://github.com/facebookresearch/lagrangian-ot/assets/707462/e5cff314-c9f8-4533-8719-9d9168802165)

```
./plot-learned-metric.py <experiment directories>
```

# Citations

If you find this repository helpful for your publications,
please consider citing [our paper](https://arxiv.org/abs/2406.00288):

```
@inproceedings{pooladian2024neural,
  title={Neural Optimal Transport with Lagrangian Costs},
  author={Pooladian, Aram-Alexandre and Domingo-Enrich, Carles and Chen, Ricky TQ and Amos, Brandon},
  booktitle={The 40th Conference on Uncertainty in Artificial Intelligence},
  years={2024}
}
```

# Licensing
This repository is licensed under the
[CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).
