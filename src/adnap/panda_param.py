"""
Parameterized version of the Panda robot.
"""
import numpy as np
from roboticstoolbox import DHRobot, RevoluteMDH
from spatialmath import base

q_min = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
q_max = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
dq_max = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
ddq_max = np.array([15, 7.5, 10, 12.5, 15, 20, 20])


def sample(num: int):
  """
  Sample from the Panda state-space.

  Args:
    n: Number of samples to draw.
  
  Returns:
    Tuple consisting of

    - q: Joint positions.
    - dq: Joint velocities.
    - ddq: Joint accelerations.
  """
  q_data = np.random.uniform(low=q_min, high=q_max, size=(num, 7))
  dq_data = np.random.uniform(low=-dq_max, high=dq_max, size=(num, 7))
  ddq_data = np.random.uniform(low=-ddq_max, high=ddq_max, size=(num, 7))
  return q_data, dq_data, ddq_data


def create_com_symbols(idx: int):
  """
  Create SymPy symbols for center of mass index `idx`.
  """
  return base.symbol(f'c_{idx}x,c_{idx}y,c_{idx}z')


def create_inertia_symbols(idx: int):
  """
  Create SymPy symbols for inertia matrix of index `idx`.
  """
  return base.symbol(
      f'I_{idx}xx,I_{idx}yy,I_{idx}zz,I_{idx}xy,I_{idx}yz,I_{idx}xz')


def unflatten_params(params: np.ndarray):
  m = params[:7]
  c = params[7:28].reshape((7, 3))
  I = params[28:].reshape((7, 6))
  return m, c, I


class PandaParameterized(DHRobot):
  """
  DH version of the Panda robot parameterized in the
  dynamic physical parameters.
  """

  def __init__(self, m, c, I, **kwargs):
    """
    Construct a Panda `DHRobot` given dynamic parametes.

    Args:
        m: Link-mass of shape (7,)
        c: Center of mass of shape (7, 3)
        I: Link-inertia of shape (7, 6) inertia indices are in
          the order :math:`I_{xx}, I_{yy}, I_{zz}, I_{xy}, I_{yz}, I_{xz}`.
    """

    # deg = np.pi/180
    mm = 1e-3
    tool_offset = (103) * mm

    flange = (107) * mm
    # d7 = (58.4)*mm

    pi = base.pi()

    # This Panda model is defined using modified
    # Denavit-Hartenberg parameters
    L = [
        RevoluteMDH(
            a=0.0,
            d=0.333,
            alpha=0.0,
            qlim=np.array([-2.8973, 2.8973]),
            m=m[0],
            I=I[0],
            r=c[0],
            G=1,
        ),
        RevoluteMDH(
            a=0.0,
            d=0.0,
            alpha=-pi / 2,
            qlim=np.array([-1.7628, 1.7628]),
            m=m[1],
            I=I[1],
            r=c[1],
            G=1,
        ),
        RevoluteMDH(
            a=0.0,
            d=0.316,
            alpha=pi / 2,
            qlim=np.array([-2.8973, 2.8973]),
            m=m[2],
            I=I[2],
            r=c[2],
            G=1,
        ),
        RevoluteMDH(
            a=0.0825,
            d=0.0,
            alpha=pi / 2,
            qlim=np.array([-3.0718, -0.0698]),
            m=m[3],
            I=I[3],
            r=c[3],
            G=1,
        ),
        RevoluteMDH(
            a=-0.0825,
            d=0.384,
            alpha=-pi / 2,
            qlim=np.array([-2.8973, 2.8973]),
            m=m[4],
            I=I[4],
            r=c[4],
            G=1,
        ),
        RevoluteMDH(
            a=0.0,
            d=0.0,
            alpha=pi / 2,
            qlim=np.array([-0.0175, 3.7525]),
            m=m[5],
            I=I[5],
            r=c[5],
            G=1,
        ),
        RevoluteMDH(
            a=0.088,
            d=flange,
            alpha=pi / 2,
            qlim=np.array([-2.8973, 2.8973]),
            m=m[6],
            I=I[6],
            r=c[6],
            G=1,
        ),
    ]

    tool = base.transl(0, 0, tool_offset) @ base.trotz(-pi / 4)

    super().__init__(L,
                     name="Panda",
                     manufacturer="Franka Emika",
                     meshdir="meshes/FRANKA-EMIKA/Panda",
                     tool=tool,
                     **kwargs)

    self.qr = np.array(
        [0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4])
    self.qz = np.zeros(7)

    self.addconfiguration("qr", self.qr)
    self.addconfiguration("qz", self.qz)


def create_sym_panda() -> PandaParameterized:
  """
  Create symbolic version of parameterized Panda.
  """
  m = base.symbol('m_:7')
  I = []
  c = []
  for i in range(7):
    I.append(create_inertia_symbols(i + 1))
    c.append(create_com_symbols(i + 1))

  return PandaParameterized(m, c, I, symbolic=True)
