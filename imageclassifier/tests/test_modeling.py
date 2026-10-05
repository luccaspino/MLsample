"""Tests for model factory contracts."""

from ic.config import Cifar10Config
from ic.modeling.cnn import build_cifar10_cnn
from ic.modeling.mlp import build_delivery_mlp


def test_delivery_mlp_has_expected_parameter_count():
    assert build_delivery_mlp().count_params() == 21


def test_cifar10_model_has_expected_input_and_output_shapes():
    model = build_cifar10_cnn(Cifar10Config())

    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 10)
