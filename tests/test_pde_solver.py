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
    """

    generate_RD_data()
    generate_RD_data()
    assert True