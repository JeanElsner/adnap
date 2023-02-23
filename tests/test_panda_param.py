import unittest
import numpy as np
from adnap import panda_param


class testParameterizedPanda(unittest.TestCase):

  def test_PandaParameterized(self):
    params = np.zeros(70)
    m, c, I = panda_param.unflatten_params(params)
    robot = panda_param.PandaParameterized(m, c, I)
    sym_robot = panda_param.create_sym_panda()

    n = 10
    q, dq, ddq = panda_param.sample(n)
    for i in range(n):
      robot.inertia(q[i])
      robot.coriolis(q[i], dq[i])
      robot.gravload(q[i])
