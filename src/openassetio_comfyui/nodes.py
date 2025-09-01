# openassetio-comfyui
# Copyright (c) 2025 The Foundry Visionmongers Ltd
# SPDX-License-Identifier: Apache-2.0
"""
OpenAssetIO nodes for ComfyUI.

Also contains a singleton class representing the OpenAssetIO host
application.
"""

import hashlib
import logging

import torch
import numpy as np

from PIL import Image, ImageOps, ImageSequence

from openassetio import EntityReference
from openassetio.utils import FileUrlPathConverter
from openassetio.log import LoggerInterface
from openassetio.access import ResolveAccess
from openassetio.hostApi import ManagerFactory, HostInterface
from openassetio.pluginSystem import (
    HybridPluginSystemManagerImplementationFactory,
    PythonPluginSystemManagerImplementationFactory,
    CppPluginSystemManagerImplementationFactory,
)
from openassetio_mediacreation.traits.content import LocatableContentTrait

import node_helpers


class _OpenAssetIOHost:
    """
    Singleton class representing the OpenAssetIO host.

    Instantiates and exposes a manager and context, and provides common
    utility methods.

    The OpenAssetIO manager must be configured via a config file,
    located via the OPENASSETIO_DEFAULT_CONFIG environment variable. See
    the OpenAssetIO documentation for more details.
    """

    _instance = None

    @classmethod
    def instance(cls) -> "_OpenAssetIOHost":
        """
        Get or create the singleton instance of the OpenAssetIO host.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    class _HostInterface(HostInterface):
        def displayName(self) -> str:
            return "ComfyUI"

        def identifier(self) -> str:
            return "org.foundry.comfyui"

        def info(self) -> dict:
            return super().info()

    class _LoggerInterface(LoggerInterface):
        def __init__(self):
            super().__init__()
            self.__logger = logging.getLogger("openassetio-comfyui")

        def log(self, severity, message) -> None:
            if severity == LoggerInterface.Severity.kDebug:
                self.__logger.debug(message)
            elif severity == LoggerInterface.Severity.kInfo:
                self.__logger.info(message)
            elif severity == LoggerInterface.Severity.kWarning:
                self.__logger.warning(message)
            elif severity == LoggerInterface.Severity.kError:
                self.__logger.error(message)
            elif severity == LoggerInterface.Severity.kCritical:
                self.__logger.critical(message)
            else:
                self.__logger.log(logging.NOTSET, message)

    def __init__(self):
        """
        Load and initialise an OpenAssetIO manager plugin.
        """
        self.__logger = self._LoggerInterface()
        # Initialise plugin system, then find and load a manager.
        self.manager = ManagerFactory.defaultManagerForInterface(
            self._HostInterface(),
            HybridPluginSystemManagerImplementationFactory(
                # Prefer C++ over Python plugins/methods.
                [
                    CppPluginSystemManagerImplementationFactory(self.__logger),
                    PythonPluginSystemManagerImplementationFactory(self.__logger),
                ],
                self.__logger,
            ),
            self.__logger,
        )
        if self.manager is None:
            raise RuntimeError(
                "Could not create an OpenAssetIO manager instance. Ensure that your OpenAssetIO"
                " configuration is correct and that the environment variable"
                " OPENASSETIO_DEFAULT_CONFIG is set to a valid configuration file."
            )
        self.__context = self.manager.createContext()
        self.__file_url_path_converter = FileUrlPathConverter()

    def resolve_to_path(self, entity_reference: str) -> str:
        """
        Resolve an OpenAssetIO entity reference to a local file path.

        OpenAssetIO will raise exceptions if the entity reference is
        invalid, if there is a problem resolving the entity, or if the
        given entity location is not a `file://` URL.

        @throw RuntimeError if the entity reference resolves
        successfully, but the entity has no location.
        """
        entity_reference = self.manager.createEntityReference(entity_reference)

        traits_data = self.manager.resolve(
            entity_reference, {LocatableContentTrait.kId}, ResolveAccess.kRead, self.__context
        )
        locatable_content_trait = LocatableContentTrait(traits_data)
        url = locatable_content_trait.getLocation()

        if url is None:
            raise RuntimeError(
                f"Failed to resolve entity reference '{entity_reference}' to a location"
            )

        return self.__file_url_path_converter.pathFromUrl(url)


class ResolveImage:
    """
    Node to resolve an image from an OpenAssetIO entity reference.

    The non-OpenAssetIO logic is largely duplicated from the built-in
    ComfyUI LoadImage node (at least as of 4f5812b9).
    """

    # Tooltip to display when hovering over the node.
    DESCRIPTION = "Resolve images from an asset manager."
    # Menu category.
    CATEGORY = "image"
    # Function to call when node is executed.
    FUNCTION = "resolve_image"
    # Node outputs.
    RETURN_TYPES = ("IMAGE", "MASK")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        """
        Input sockets and widgets.
        """
        return {
            "required": {
                "entity_reference": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "placeholder": "Entity reference...",
                        "tooltip": "The entity to resolve",
                    },
                )
            }
        }

    @classmethod
    def IS_CHANGED(cls, entity_reference: str) -> str:
        """
        Resolve the entity reference to a file path, then return a hash
        of the file contents.

        This allows ComfyUI to determine if the node needs to be
        re-executed.

        Non-OpenAssetIO logic largely duplicated from the built-in
        ComfyUI LoadImage node (at least as of 4f5812b9).
        """
        image_path = _OpenAssetIOHost.instance().resolve_to_path(entity_reference)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, entity_reference: str) -> bool:
        """
        Validate that the input entity reference is valid syntax for the
        current manager.
        """
        return _OpenAssetIOHost.instance().manager.isEntityReferenceString(entity_reference)

    def resolve_image(self, entity_reference: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Resolve an OpenAssetIO entity reference to a file path, then
        load the image(s) at that path and return as tensors.

        Non-OpenAssetIO logic largely duplicated from the built-in
        ComfyUI LoadImage node (at least as of 4f5812b9).
        """
        image_path = _OpenAssetIOHost.instance().resolve_to_path(entity_reference)

        sequence = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ["MPO"]

        for frame in ImageSequence.Iterator(sequence):
            frame = node_helpers.pillow(ImageOps.exif_transpose, frame)  # noqa: PLW2901 - frame overwrite

            if frame.mode == "I":
                frame = frame.point(lambda px: px * (1 / 255))  # noqa: PLW2901 - frame overwrite
            image = frame.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if "A" in frame.getbands():
                mask = np.array(frame.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            elif frame.mode == "P" and "transparency" in frame.info:
                mask = np.array(frame.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and sequence.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return output_image, output_mask


# Plugin registration: node classes.
NODE_CLASS_MAPPINGS = {
    "OpenAssetIOResolveImage": ResolveImage,
}

# Plugin registration: node names.
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAssetIOResolveImage": "OpenAssetIO Resolve Image",
}
