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

# Quick start

$$ V_{n}(R)=\frac{\pi^{\frac{n}{2}}}{\Gamma\left(\frac{n}{2}+1\right)} R^{n} $$

```rball``` is a utility to store and interpolate instrument response matrices on the sky. 


```python
from rball import ResponseDatabase
from rball.utils import get_path_of_data_file

import h5py


%matplotlib inline
```

## Creating a ResponseDatabase object

First we will use a demo database stored in an HDF5 file to create a response database.

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

## Examining the sky grid

We can view the grid in 3D. When a point in the sky is selected, a [Delaunay triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation) is used to find the three matrices surrounding this point.

```python
rsp_db.plot_verticies_ipv(selected_location=[5,-45])
```

## Using and interpolating responses 


Internally, the ```ResponseDatabase``` is storing a 3ML ```InstrumentResponse``` object

```python
rsp_db.current_response.plot_matrix();
```

When we want to interpolate a matrix to a point on the sky, the encapsulating matricies ($M_i$) are found and the barycentered distances ($b_i$) withing the triangle is used to create an interpolated matrix ($M_{\mathrm{intrp}}$) such that

$$ M_{\mathrm{intrp}} = \sum^{3}_{i} b_i M_{i}  $$

```python
rsp_db.interpolate_to_position(0.,0)
rsp_db.current_response.plot_matrix();
```

```python
for theta in range(-90,90,30):
    rsp_db.interpolate_to_position(0., theta)
    rsp_db.current_response.plot_matrix();
```

## General notes on units and construction

The theta and phi coordinates are internally in **radians**, but all inputs are in **degrees** for the user. Moreover, both the plotting and interpolation assume inputs in J2000 RA, Dec coordinates. When subclassing ```ResponseDatabase```, an private function should be defined (``` _transform_to_instrument_coordinates```) which takes as inputs RA and Dec in degrees and transforms to the spacecraft coordinates of the response grid points resulting in a return of a theta, phi tuple in radians.



