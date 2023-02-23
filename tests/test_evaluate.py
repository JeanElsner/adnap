from argparse import Namespace
import unittest
from unittest import mock
import numpy as np
from adnap import evaluate


class testEvaluate(unittest.TestCase):

  @mock.patch('argparse.ArgumentParser.parse_args',
              return_value=Namespace(file='tests/params.npy', n=1))
  @mock.patch('os.environ.get', return_value='TEST')
  @mock.patch('panda_model.Model')
  def test_run(self, Model, *args):
    Model.return_value.mass.return_value = np.zeros((7, 7))
    Model.return_value.coriolis.return_value = np.zeros(7)
    Model.return_value.gravity.return_value = np.zeros(7)
    evaluate.run()
    Model.assert_called_once_with('TEST')
    Model.return_value.mass.assert_called_once()
    Model.return_value.coriolis.assert_called_once()
    Model.return_value.gravity.assert_called_once()
