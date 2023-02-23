# ADNAP
Reverse engineer the Panda dynamics model.

[![build](https://img.shields.io/github/actions/workflow/status/JeanElsner/adnap/python-test.yml)](https://github.com/JeanElsner/adnap/actions/workflows/python-test.yml)
[![readthedocs](https://img.shields.io/readthedocs/adnap)](https://adnap.readthedocs.io/)
[![pypi-version](https://img.shields.io/pypi/v/adnap)](https://pypi.org/project/adnap/)
[![license](https://img.shields.io/github/license/JeanElsner/adnap)](https://github.com/JeanElsner/adnap/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/JeanElsner/adnap/branch/main/graph/badge.svg?token=6GOKVDXZJ9)](https://codecov.io/gh/JeanElsner/adnap)
[![pylint](https://jeanelsner.github.io/adnap/pylint.svg)](https://jeanelsner.github.io/adnap/pylint.log)
[![pypi](https://img.shields.io/pypi/pyversions/adnap)](https://pypi.org/project/adnap/)

## Install
```
pip install adnap
```
### Requirements
The dependency `panda-model` requires `POCO C++ libraries` and `Eigen3` to be installed. On Ubuntu install them by running:
```
sudo apt-get install libpoco-dev libeigen3-dev
```

## Usage
Point the environment variable to the libfranka shared library downloaded with [`panda-model`](https://github.com/JeanElsner/panda_model)
```
export PANDA_MODEL_PATH=<path-to-libfrankamodel.so>
```
Run optimization with 10 random samples from the Panda state-space and save results in params.npy:
```
adnap-optimize -n 10 -o params.npy
```
Evaluate the optimized physical parameters against the shared library on 1000 random samples:
```
adnap-evaluate -n 1000 params.npy
```
