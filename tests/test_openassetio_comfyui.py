# openassetio-comfyui
# Copyright (c) 2025 The Foundry Visionmongers Ltd
# SPDX-License-Identifier: Apache-2.0
"""
Tests for `openassetio-comfyui` package.
"""

import contextlib
import hashlib
# D101: "Missing docstring in public class"
# D102: "Missing docstring in public method"
# ruff: noqa: D101,D102

import inspect
import pathlib

import numpy as np
import pytest
import torch
from PIL import Image

from openassetio_comfyui.nodes import ResolveImage, _OpenAssetIOHost


class Test_ResolveImage_init:
    """
    Test that the node can be instantiated.
    """

    def test_has_correct_type(self, resolve_image_node):
        assert isinstance(resolve_image_node, ResolveImage)


class Test_ResolveImage_constants:
    """
    Test the node's metadata.
    """

    def test_has_expected_values(self):
        assert ResolveImage.DESCRIPTION == "Resolve images from an asset manager."
        assert ResolveImage.CATEGORY == "image"
        assert ResolveImage.FUNCTION == "resolve_image"
        assert ResolveImage.RETURN_TYPES == ("IMAGE", "MASK")

    def test_function_matches(self):
        assert inspect.isfunction(getattr(ResolveImage, ResolveImage.FUNCTION))


class Test_ResolveImage_INPUT_TYPES:
    def test_has_expected_structure(self):
        input_types = ResolveImage.INPUT_TYPES()

        assert input_types == {
            "required": {
                "entity_reference": (
                    "STRING",
                    {
                        "multiline": False,
                        "placeholder": "Entity reference...",
                        "default": "",
                        "tooltip": "The entity to resolve",
                    },
                )
            }
        }


class Test_ResolveImage_VALIDATE_INPUTS:
    def test_when_is_a_reference_then_returns_true(self):
        valid = ResolveImage.VALIDATE_INPUTS("bal:///")
        assert valid

    def test_when_not_a_reference_then_returns_false(self):
        valid = ResolveImage.VALIDATE_INPUTS("notbal:///")
        assert not valid

    def test_when_no_manager_found_then_raises(self, assert_raises_missing_manager):
        with assert_raises_missing_manager():
            ResolveImage.VALIDATE_INPUTS("bal:///")


class Test_ResolveImage_IS_CHANGED:
    def test_when_reference_resolves_to_file_then_hash_matches_file(self, image_file_hash):
        actual_file_hash = ResolveImage.IS_CHANGED("bal:///image_file")

        assert actual_file_hash == image_file_hash

    def test_when_reference_has_no_path_then_raises(self, image_file_hash):
        expected_error_message = (
            "Failed to resolve entity reference 'bal:///no_file' to a location"
        )

        with pytest.raises(RuntimeError, match=expected_error_message):
            ResolveImage.IS_CHANGED("bal:///no_file")

    def test_when_no_manager_found_then_raises(self, assert_raises_missing_manager):
        with assert_raises_missing_manager():
            ResolveImage.IS_CHANGED("bal:///image_file")


class Test_ResolveImage_resolve_image:
    def test_when_reference_resolves_to_file_then_returns_image(
        self, resolve_image_node, image_file_path
    ):
        images, masks = resolve_image_node.resolve_image("bal:///image_file")

        assert isinstance(images, torch.Tensor)
        assert images.shape == (1, 2, 2, 3)
        assert isinstance(masks, torch.Tensor)
        assert masks.shape == (1, 2, 2)

        image = images[0]
        mask = masks[0]

        # Assert channels at each pixel.
        assert torch.equal(image[0, 0], torch.tensor([0, 0, 0]))
        assert torch.equal(image[0, 1], torch.tensor([1, 1, 1]))
        assert torch.equal(image[1, 0], torch.tensor([0, 0, 0]))
        assert torch.equal(image[1, 1], torch.tensor([1, 1, 1]))

        # Note: mask is `1 - alpha`.
        assert mask[0, 0].item() == 0
        assert mask[0, 1].item() == 1
        assert mask[1, 0].item() == 1
        assert mask[1, 1].item() == 0

    def test_when_no_manager_found_then_raises(
        self, resolve_image_node, assert_raises_missing_manager
    ):
        with assert_raises_missing_manager():
            resolve_image_node.resolve_image("bal:///image_file")


@pytest.fixture
def assert_raises_missing_manager(monkeypatch):
    """
    Fixture to assert that a RuntimeError is raised when no manager can
    be created.

    Since a lazily-created singleton is used to manage the OpenAssetIO
    host, this assertion must be tested for every function that uses
    that singleton.
    """

    @contextlib.contextmanager
    def assert_ctx():
        monkeypatch.delenv("OPENASSETIO_DEFAULT_CONFIG", raising=False)
        expected_error_msg = (
            "Could not create an OpenAssetIO manager instance. Ensure that your OpenAssetIO"
            " configuration is correct and that the environment variable"
            " OPENASSETIO_DEFAULT_CONFIG is set to a valid configuration file."
        )
        with pytest.raises(RuntimeError, match=expected_error_msg):
            yield

    return assert_ctx


@pytest.fixture
def image_file_hash(image_file_path):
    """
    Fixture to provide the hash of the temporary image file.
    """
    hasher = hashlib.sha256()
    hasher.update(image_file_path.read_bytes())
    return hasher.hexdigest()


@pytest.fixture
def image_file_path(monkeypatch, tmp_path):
    """
    Fixture to provide a temporary image file.
    """
    # See bal_db.json - the resolved path will interpolate this env var.
    monkeypatch.setenv("test_tmp_dir", str(tmp_path))
    # See bal_db.json - the resolved path for bal:///cat
    file_path = pathlib.Path(tmp_path) / "cat.png"

    content = np.array(
        [[[0, 0, 0, 255], [255, 255, 255, 0]], [[0, 0, 0, 0], [255, 255, 255, 255]]]
    ).astype(np.uint8)
    img = Image.fromarray(content)
    img.save(file_path)
    return file_path


@pytest.fixture
def resolve_image_node():
    """
    Fixture to create an ResolveImage node instance.
    """
    return ResolveImage()


@pytest.fixture(autouse=True)
def openassetio_config_env_var(monkeypatch, resources_dir):
    """
    Fixture to set the OPENASSETIO_DEFAULT_CONFIG environment variable
    to point to a test config file.
    """
    test_config_path = resources_dir / "openassetio_config.toml"
    monkeypatch.setenv("OPENASSETIO_DEFAULT_CONFIG", str(test_config_path))


@pytest.fixture(autouse=True)
def reset_singleton_host(monkeypatch):
    """
    Fixture to reset the singleton instance before each test.
    """
    _OpenAssetIOHost._instance = None


@pytest.fixture(scope="module")
def resources_dir():
    """
    Fixture to provide the path to the resources directory.
    """
    return pathlib.Path(__file__).parent / "resources"
