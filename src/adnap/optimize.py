"""
Optimize dynamic parmeters to match libfranka dynamics.
"""
import argparse
import os
from typing import Callable

import numpy as np
import panda_model
from scipy.optimize import least_squares

from .panda_param import PandaParameterized, sample


def coriolis_error(model: panda_model.Model, opt_model: PandaParameterized,
                   q_data: np.ndarray, dq_data: np.ndarray):
  """
  Computes model error in coriolis term.
  """
  return opt_model.coriolis(q_data, dq_data) @ dq_data - model.coriolis(
      q_data, dq_data, np.zeros((3, 3)), 0, np.zeros(3))


inertia_idx = np.tril_indices(7)


def mass_error(model: panda_model.Model, opt_model: PandaParameterized,
               q_data: np.ndarray):
  """
  Computes model error in inertia matrix.
  """
  return model.mass(
      q_data, np.zeros((3, 3)), 0,
      np.zeros(3))[inertia_idx] - opt_model.inertia(q_data)[inertia_idx]


def gravity_error(model: panda_model.Model, opt_model: PandaParameterized,
                  q_data: np.ndarray):
  """
  Computes model error in the gravity term.
  """
  return opt_model.gravload(q_data) - model.gravity(q_data, 0, np.zeros(3))


def make_residual(num_samples: int,
                  lib_path: str) -> Callable[[np.ndarray], np.ndarray]:
  """
  Create a residual function for least squares optimization
  that uses `num_samples` samples.

  Returns:
    Function that maps 70 physical parameters to residual model errors
    with shape (20,).
  """
  q_data, dq_data, __ = sample(num_samples)

  def residual(params):
    """
    Computes residuals for least square optimizaiton.

    Args:
      params: Flattened parameters of shape (70,).

    Returns:
      numpy.ndarray: Residuals of shape (20,).
    """
    m = params[:7]
    c = params[7:28].reshape((7, 3))
    I = params[28:].reshape((7, 6))

    model = panda_model.Model(lib_path)
    opt_model = PandaParameterized(m, c, I)

    mass_res = []
    coriolis_res = []
    gravity_res = []

    for i in range(num_samples):
      mass_res.append(mass_error(model, opt_model, q_data[i]))
      coriolis_res.append(
          coriolis_error(model, opt_model, q_data[i], dq_data[i]))
      gravity_res.append(gravity_error(model, opt_model, q_data[i]))

    return np.concatenate(
        (np.array(mass_res).flatten(), np.array(coriolis_res).flatten(),
         np.array(gravity_res).flatten()))

  return residual


def run():
  """
  Entrypoint to run optimization with random initialization.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('-n',
                      help="Number of samples used in residual.",
                      default=10,
                      type=int)
  parser.add_argument('--max-nfev',
                      help='Maximum number of function evaluations.',
                      type=int,
                      default=None)
  parser.add_argument('-o',
                      '--output',
                      help='Outpuf file to write parameters into',
                      type=str,
                      required=True)
  args = parser.parse_args()

  lower_bounds = np.zeros(70)
  lower_bounds[:7] = 0  # mass > 0kg
  lower_bounds[7:] = -.15  # distance of com from joint axis <= 15cm
  lower_bounds[28:] = -1  # inertia elements > -1

  # diagonal inertia elements > 0
  lower_bounds[28:31] = 0
  lower_bounds[34:37] = 0
  lower_bounds[40:43] = 0
  lower_bounds[46:49] = 0
  lower_bounds[52:53] = 0
  lower_bounds[58:61] = 0
  lower_bounds[64:67] = 0

  upper_bounds = np.zeros(70)
  upper_bounds[:7] = 10  # mass < 10kg
  upper_bounds[7:] = .15  # distance of com from joint axis <= 15cm
  upper_bounds[28:] = 1  # inertia elements < 1

  x_start = np.random.uniform(lower_bounds, upper_bounds)

  libfranka = os.environ.get('PANDA_MODEL_PATH')
  if libfranka is None:
    raise RuntimeError('Please set environment variable PANDA_MODEL_PATH ' +
                       'to the shared library downloaded with panda-model.')

  res = least_squares(make_residual(args.n, libfranka),
                      x_start,
                      bounds=(lower_bounds, upper_bounds),
                      verbose=2,
                      max_nfev=args.max_nfev)
  np.save(args.output, res.x)
  print(res)
