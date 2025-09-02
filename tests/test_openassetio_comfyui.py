#!/usr/bin/env python
# openassetio-comfyui
# Copyright (c) 2025 The Foundry Visionmongers Ltd
# SPDX-License-Identifier: Apache-2.0

"""Tests for `openassetio-comfyui` package."""

import pytest
from src.openassetio_comfyui.nodes import Example


@pytest.fixture
def example_node():
    """Fixture to create an Example node instance."""
    return Example()


def test_example_node_initialization(example_node):
    """Test that the node can be instantiated."""
    assert isinstance(example_node, Example)


def test_return_types():
    """Test the node's metadata."""
    assert Example.RETURN_TYPES == ("IMAGE",)
    assert Example.FUNCTION == "test"
    assert Example.CATEGORY == "Example"
