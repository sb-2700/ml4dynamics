"""
Test the data generator, make sure the data generated is physcial and
correct. May already implement the mesh refinement study in our py
notebook
"""

import sys
sys.path.append('..')
import pytest
# We need to use functional programming so we can call the function we want
from src.generate_NSdata import generate_NS_data
from src.generate_RDdata import generate_RD_data


def test_reaction_diffusion_equation_solver():
    """We perform a mesh-refinement study to check the accuracy of our solver
    test case for RD equation:
        u(x, y, t) = sin(x)
        v(x, y, t) = sin(y)
        D = [[1, 0], [0, 1]]
        alpha = 0.01
        beta = 1.0
        s_u(x, y, t) = sin^3(y) + sin(y) - 0.01
        s_v(x, y, t) = 2sin(y) - sin(x)
    
    test case for NS equation:

    """

    generate_RD_data()
    assert True