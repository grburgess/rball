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

<!-- #region -->
# Localization with RBallLike

`rball` provides a 3ML plugin that can perform localization of point sources.



First we need to read in the database.
<!-- #endregion -->

```python
from astromodels import Powerlaw, PointSource, Model, Log_uniform_prior, Uniform_prior
from threeML import BayesianAnalysis, DataList


from rball import ResponseDatabase, RBallLike
from rball.utils import get_path_of_data_file
import h5py

%matplotlib inline
```

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

We can create an RBallLike from normal PHA files. We will use a simulated spectrum that comes from a position on the sky (RA: 150 Dec: 0) with a power law spectrum.

```python
demo_plugin = RBallLike.from_ogip("demo", 
                                  observation=get_path_of_data_file("demo.pha"),
                                  spectrum_number=1,
                                  response_database=rsp_db)
```

## Fitting for the localization

We will create a 3ML point source and assign priors to the spectral parameters. The ```RBallLike``` plugin automatically assigns uniform spherical priors to the sky position. This can always be altered in the model



```python
source_function = Powerlaw(K=1, index=-2, piv=100)

source_function.K.prior = Log_uniform_prior(lower_bound=1e-1, upper_bound=1e1)
source_function.index.prior = Uniform_prior(lower_bound=-4, upper_bound=0)

ps = PointSource("ps", 150.,1., spectral_shape=source_function)

model = Model(ps)
```

```python
ba = BayesianAnalysis(model, DataList(demo_plugin))
```

```python
model
```

Now we can sample the spectrum and position to do the localization.

```python
ba.set_sampler('emcee')
ba.sampler.setup(n_walkers=50, n_iterations=1000., n_burnin=1000)
```

```python
ba.sample();
```

```python
ba.results.corner_plot();
```

```python

```
