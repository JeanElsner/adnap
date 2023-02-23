"""
Evaluate dynamic parmeters against libfranka dynamics.
"""
import argparse
import os

import numpy as np
import panda_model

from .optimize import coriolis_error, gravity_error, mass_error
from .panda_param import PandaParameterized, sample, unflatten_params


def run():
  """
  Evaluate physical parameters from a file against model from libfranka
  shared library on random samples drawn from the robot's state space.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('-n',
                      help="Number of samples used in evaluation.",
                      default=1000,
                      type=int)
  parser.add_argument('file',
                      help='Input file containing optimized parameters.',
                      type=str)
  args = parser.parse_args()

  libfranka = os.environ.get('PANDA_MODEL_PATH')
  if libfranka is None:
    raise RuntimeError('Please set environment variable PANDA_MODEL_PATH ' +
                       'to the shared library downloaded with panda-model.')

  m, c, I = unflatten_params(np.load(args.file))
  opt_model = PandaParameterized(m, c, I)
  model = panda_model.Model(libfranka)

  q_data, dq_data, __ = sample(args.n)
  coriolis = []
  mass = []
  gravity = []
  for i in range(args.n):
    coriolis.append(
        np.abs(coriolis_error(model, opt_model, q_data[i], dq_data[i])))
    mass.append(np.abs(mass_error(model, opt_model, q_data[i])))
    gravity.append(np.abs(gravity_error(model, opt_model, q_data[i])))

  print('inertia_error:', np.mean(np.array(mass).flatten(), axis=0))
  print('coriolis_error:', np.mean(np.array(coriolis).flatten(), axis=0))
  print('gravity_error:', np.mean(np.array(gravity).flatten(), axis=0))

  print('max inertia_error:', np.max(np.array(mass).flatten()))
  print('max coriolis_error:', np.max(np.array(coriolis).flatten()))
  print('max gravity_error:', np.max(np.array(gravity).flatten()))
