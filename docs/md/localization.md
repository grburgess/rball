---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Localization with RBallLike

```rball``` provides a 3ML plugin that can perform localization of point sources.

```python
from astromodels import Powerlaw, PointSource, Model, Log_uniform_prior, Uniform_prior
from threeML import BayesianAnalysis, DispersionSpectrumLike, OGIPLike, DataList


from rball import ResponseDatabase, RBallLike
from rball.utils import get_path_of_data_file
import h5py

%matplotlib notebook
```

First we need to read in the database.

```python
file_name = get_path_of_data_file("demo_rsp_database.h5")

with h5py.File(file_name, "r") as f:

    
    # the base grid point matrices
    # should be an (N grid points, N ebounds, N monte carlo energies)
    # numpy array

    list_of_matrices = f["matrix"][()]
    
    # theta and phi are the 
    # lon and lat points of 
    # the matrix database in radian
    
    theta = f["theta"][()]

    phi = f["phi"][()]

    # the bounds of the response
    
    ebounds = f["ebounds"][()]

    mc_energies = f["mc_energies"][()]

    rsp_db = ResponseDatabase(
        list_of_matrices=list_of_matrices,
        theta=theta,
        phi=phi,
        ebounds=ebounds,
        monte_carlo_energies=mc_energies,
    )


```

We can create an RBallLike from normal PHA files. We will use a simulated spectrum that comes from a position on the sky 

```python
demo_plugin = RBallLike.from_ogip("demo", 
                                  observation=get_path_of_data_file("demo.pha"),
                                  spectrum_number=1,
                                  response_database=rsp_db)
```

```python
source_function = Powerlaw(K=1, index=-2, piv=100)


```

```python

```

```python
source_function.K.prior = Log_uniform_prior(lower_bound=1e-1, upper_bound=1e1)
source_function.index.prior = Uniform_prior(lower_bound=-4, upper_bound=0)

ps = PointSource("ps", 0,0, spectral_shape=source_function)

model = Model(ps)
```

```python
ba = BayesianAnalysis(model, DataList(demo_plugin))
```

```python
ps.position.ra = 150
ps.dec = 0
```

```python
demo_plugin.get_model()

rsp_db.current_sky_position
```

```python
ba.set_sampler('multinest')
```

```python
ba.sampler.setup(n_live_points=400)
```

```python
ba.sample();
```

```python
ba.results.corner_plot();
```

```python
OGIPLikeIPLikeIPLikeIPLikeIPLikePLikePLike?
```

```python
xxx= demo_plugin.get_simulated_dataset()
```

```python
%debug
```

```python
demo_plugin._observed_spectrum
```

```python

```
