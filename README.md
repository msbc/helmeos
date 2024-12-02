# helmeos

[![codecov](https://codecov.io/github/msbc/helmeos/graph/badge.svg?token=9C1XFJO2SQ)](https://codecov.io/github/msbc/helmeos)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit)

A Python implementation/port of [Frank Timmes' Helmholtz EoS](http://cococubed.com/code_pages/eos.shtml).

This package provides a simple interface to the Helmholtz EoS for a wide range of temperatures and densities. The Helmholtz EoS is a tabulated equation of state for stellar matter, which is based on the Helmholtz free energy. The Helmholtz free energy is a function of temperature, density, and composition, and is used to calculate the pressure, internal energy, and entropy of a gas. This package does not provide a way to calculate the interpolation tables, but it does provide a way to read the tables and interpolate the values.

## References

- [Timmes, F. X., & Arnett, D. 1999, ApJS, 125, 277](https://ui.adsabs.harvard.edu/abs/1999ApJS..125..277T/abstract)
- [Timmes, F. X., & Douglas, S. 2000, ApJS, 126, 501](https://ui.adsabs.harvard.edu/abs/2000ApJS..126..501T/abstract)

## Prerequisites

Python packages:

- numpy
- matplotlib (optional, for plotting)

This is only tested on Python3.

## Installation

This package is available on PyPI, so you can install it using pip:
```pip install helmeos```

This package is also available on conda-forge, so you can install it using conda:
```conda install -c conda-forge helmeos```

## Example

See [example.py](helmeos/example.py) for example code.
