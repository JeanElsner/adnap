import unittest
import numpy as np
from adnap import panda_param


class testParameterizedPanda(unittest.TestCase):

  def test_Panda(self):
    params = np.zeros(70)
    m, c, I = panda_param.unflatten_params(params)
    panda_param.PandaParameterized(m, c, I)
    panda_param.create_sym_panda()
