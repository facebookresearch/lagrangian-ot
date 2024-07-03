# [Neural Optimal Transport with Lagrangian Costs](https://openreview.net/forum?id=myb0FKB8C9).

[Aram-Alexandre Pooladian](https://arampooladian.com/),
[Carles Domingo-Enrich](https://cdenrich.github.io/),
[Ricky T. Q. Chen](https://rtqichen.github.io/), and
[Brandon Amos](https://bamos.github.io/)

> We investigate the optimal transport problem between probability measures when the underlying cost function is understood to satisfy a least action principle, also known as a Lagrangian cost. These generalizations are useful when connecting observations from a physical system where the transport dynamics are influenced by the geometry of the system, such as obstacles (e.g., incorporating barrier functions in the Lagrangian), and allows practitioners to incorporate a priori knowledge of the underlying system such as non-Euclidean geometries (e.g., paths must be circular). Our contributions are of computational interest, where we demonstrate the ability to efficiently compute geodesics and amortize spline-based paths, which has not been done before, even in low dimensional problems. Unlike prior work, we also output the resulting Lagrangian optimal transport map without requiring an ODE solver. We demonstrate the effectiveness of our formulation on low-dimensional examples taken from prior work.

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

![](https://private-user-images.githubusercontent.com/707462/345490833-9fb0b16f-8e38-4417-81b2-dfa1f67868ec.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjAwMjA4ODIsIm5iZiI6MTcyMDAyMDU4MiwicGF0aCI6Ii83MDc0NjIvMzQ1NDkwODMzLTlmYjBiMTZmLThlMzgtNDQxNy04MWIyLWRmYTFmNjc4NjhlYy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNzAzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDcwM1QxNTI5NDJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1iNjBlYzU1ZjdiZTU4NzViZDM0M2Y3M2Q0YzY2ZDhiMjBjZjY1NjMyMTIzMjUzZDQzZTFlZWNhNzUxMzUwNTM3JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.u_suEf8xCZE3P8hFdJvW1r5_51V873R7v9LkrHCjTxg)

### Figure 1

![](https://private-user-images.githubusercontent.com/707462/345490574-d2894e8f-67c2-42d3-828b-d64d8ba6161b.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjAwMjA4ODIsIm5iZiI6MTcyMDAyMDU4MiwicGF0aCI6Ii83MDc0NjIvMzQ1NDkwNTc0LWQyODk0ZThmLTY3YzItNDJkMy04MjhiLWQ2NGQ4YmE2MTYxYi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNzAzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDcwM1QxNTI5NDJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT02YmUxMTkzNTZhMmFlN2IzYzM1OTM0YmU5MzNmOTA5ZmM3ZGNmM2U0Yjc4OGUyM2UxM2NjYjZhNTFhZTMzNGJkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.4-aVL0dgeDCuawN-8dMqdpZxW3ipT7M6AzUSk1vgAmU)


```
./train_ot.py geometry=gsb_gmm
./train_ot.py geometry=scarvelis_circle
```

### Figure 2
![](https://private-user-images.githubusercontent.com/707462/345491019-f3236e4b-15c9-45db-9a8e-63460d751299.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjAwMjA4ODIsIm5iZiI6MTcyMDAyMDU4MiwicGF0aCI6Ii83MDc0NjIvMzQ1NDkxMDE5LWYzMjM2ZTRiLTE1YzktNDVkYi05YThlLTYzNDYwZDc1MTI5OS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNzAzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDcwM1QxNTI5NDJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0xNTM2ZDE1MThiODk4Y2FkZDhlNzhkNWQ1Njk5ZmQ2OGI0NzkyNDA2ZDhmMmJhYzk3NTM3ODRjYTkwMTBiZDZmJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.UfheTMPN05y3zCb7L0Z65PntRHnj-Tfis7NjWtttKhs)

```
./train_ot.py geometry=lsb_box
./train_ot.py geometry=lsb_slit
./train_ot.py geometry=lsb_hill
./train_ot.py geometry=lsb_well
```

## Metric learning with NLOT
![](https://private-user-images.githubusercontent.com/707462/345491132-8cda8217-efa1-41b4-9b1a-810ebb60c1f4.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjAwMjA4ODIsIm5iZiI6MTcyMDAyMDU4MiwicGF0aCI6Ii83MDc0NjIvMzQ1NDkxMTMyLThjZGE4MjE3LWVmYTEtNDFiNC05YjFhLTgxMGViYjYwYzFmNC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNzAzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDcwM1QxNTI5NDJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT03OWJjZDI0YWI1MzJmZjkzODcyYjY3OWQ0YmU5MjkxNzk4MDM3Njc2ZGI5MjU0MmUwOGQ0ZTYxM2Y0NzEzNjVmJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.QZWHtemXmt5APf-4vWHuExw75u4BEP4AjFCnwtpKfT0)

### Table 2
![](https://private-user-images.githubusercontent.com/707462/345493436-8e263d77-3979-495b-b483-d5cce90011ef.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjAwMjExMDAsIm5iZiI6MTcyMDAyMDgwMCwicGF0aCI6Ii83MDc0NjIvMzQ1NDkzNDM2LThlMjYzZDc3LTM5NzktNDk1Yi1iNDgzLWQ1Y2NlOTAwMTFlZi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNzAzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDcwM1QxNTMzMjBaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05OWM2YThjOTUyOWNmYzk0NTM4OWJmNzVkMzRjM2Y3ZDUxOTc3NjEyNTBjOTllMDBjYWVmNjU2MmZlZWU0M2QyJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.DTmEWLN4S5dwoLZgH-jbveqG32EuF-7QO74gYpQSViM)

```
./train_ot_scarvelis.py geometry=scarvelis_circle
./train_ot_scarvelis.py geometry=scarvelis_xpath
./train_ot_scarvelis.py geometry=scarvelis_vee
```

### Figures 3 and 4
![](https://private-user-images.githubusercontent.com/707462/345493614-bef79e05-9d95-4bc9-9495-dbd41849779f.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjAwMjExMDAsIm5iZiI6MTcyMDAyMDgwMCwicGF0aCI6Ii83MDc0NjIvMzQ1NDkzNjE0LWJlZjc5ZTA1LTlkOTUtNGJjOS05NDk1LWRiZDQxODQ5Nzc5Zi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNzAzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDcwM1QxNTMzMjBaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT02MTYwYjQ5YmFjMzE2ZTA5M2MyYzMyNDkxNzlmODVjZWVmMjVhMjEzNGZkMjRjNmMzMTQ4MGM4M2YwNWQ2NzZjJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.0hUPZDdkvvafKP8G2su-9w4kRGknmgaG9sfZEXkNn7o)

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
