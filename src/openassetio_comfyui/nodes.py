# openassetio-comfyui
# Copyright (c) 2025 The Foundry Visionmongers Ltd
# SPDX-License-Identifier: Apache-2.0
"""
OpenAssetIO nodes for ComfyUI.

Also contains a singleton class representing the OpenAssetIO host
application.
"""

import hashlib
import json
import logging
import pathlib
import shutil

import torch
import numpy as np

from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo

from openassetio import EntityReference
from openassetio.utils import FileUrlPathConverter
from openassetio.log import LoggerInterface
from openassetio.access import ResolveAccess, PublishingAccess
from openassetio.hostApi import ManagerFactory, HostInterface
from openassetio.pluginSystem import (
    HybridPluginSystemManagerImplementationFactory,
    PythonPluginSystemManagerImplementationFactory,
    CppPluginSystemManagerImplementationFactory,
)
from openassetio_mediacreation.specifications.twoDimensional import (
    PlanarBitmapImageResourceSpecification,
)
from openassetio_mediacreation.traits.content import LocatableContentTrait

import folder_paths
import node_helpers
from comfy.cli_args import args


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
        self.logger = self._LoggerInterface()
        # Initialise plugin system, then find and load a manager.
        self.manager = ManagerFactory.defaultManagerForInterface(
            self._HostInterface(),
            HybridPluginSystemManagerImplementationFactory(
                # Prefer C++ over Python plugins/methods.
                [
                    CppPluginSystemManagerImplementationFactory(self.logger),
                    PythonPluginSystemManagerImplementationFactory(self.logger),
                ],
                self.logger,
            ),
            self.logger,
        )
        if self.manager is None:
            raise RuntimeError(
                "Could not create an OpenAssetIO manager instance. Ensure that your OpenAssetIO"
                " configuration is correct and that the environment variable"
                " OPENASSETIO_DEFAULT_CONFIG is set to a valid configuration file."
            )
        self.context = self.manager.createContext()
        self.file_url_path_converter = FileUrlPathConverter()

    def resolve_to_path(
        self,
        entity_reference: EntityReference | str,
        access_mode: ResolveAccess = ResolveAccess.kRead,
    ) -> str:
        """
        Resolve an OpenAssetIO entity reference to a local file path.

        OpenAssetIO will raise exceptions if the entity reference is
        invalid, if there is a problem resolving the entity, or if the
        given entity location is not a `file://` URL.

        @throw RuntimeError if the entity reference resolves
        successfully, but the entity has no location.
        """
        if isinstance(entity_reference, str):
            entity_reference = self.manager.createEntityReference(entity_reference)

        traits_data = self.manager.resolve(
            entity_reference, {LocatableContentTrait.kId}, access_mode, self.context
        )
        locatable_content_trait = LocatableContentTrait(traits_data)
        url = locatable_content_trait.getLocation()

        if url is None:
            raise RuntimeError(
                f"Failed to resolve entity reference '{entity_reference}' to a location"
            )

        return self.file_url_path_converter.pathFromUrl(url)


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


class PublishImage:
    """
    Node to publish an image to an OpenAssetIO entity reference.

    The non-OpenAssetIO logic is largely duplicated from the built-in
    ComfyUI SaveImage node (at least as of 4f5812b9).
    """

    # Tooltip to display when hovering over the node.
    DESCRIPTION = "Publishes images to an asset manager."
    # Menu category.
    CATEGORY = "image"
    # Function to call when node is executed.
    FUNCTION = "publish_images"
    # Node outputs.
    RETURN_TYPES = ()
    # Marks this node as a terminal node, ensuring the associated
    # subgraph is executed when running the graph.
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        """
        Input sockets and widgets.
        """
        return {
            "required": {
                "entity_reference": (
                    "STRING",
                    {"multiline": False, "tooltip": "The entity to publish to"},
                ),
                "images": ("IMAGE", {"tooltip": "The images to save."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, entity_reference: str) -> bool:
        """
        Validate that the input entity reference is valid syntax for the
        current manager.
        """
        return _OpenAssetIOHost.instance().manager.isEntityReferenceString(entity_reference)

    def __init__(self):
        """
        Initialise node.

        Duplicated from SaveImage node.
        """
        self.compress_level = 4

    def publish_images(
        self, entity_reference: str, images: torch.Tensor, prompt=None, extra_pnginfo=None
    ) -> dict:
        """
        Publish the input images to the specified OpenAssetIO entity.

        Assumes the working reference provided by `preflight()` can be
        resolved to a destination file path to write to.

        Non-OpenAssetIO logic largely duplicated from the built-in
        ComfyUI SaveImage node (at least as of 4f5812b9).

        In particular, we inherit support for a "%batch_num%"
        placeholder in the resolved path.
        """
        entity_reference = _OpenAssetIOHost.instance().manager.createEntityReference(
            entity_reference
        )

        spec = PlanarBitmapImageResourceSpecification.create()

        working_ref = _OpenAssetIOHost.instance().manager.preflight(
            entity_reference,
            spec.traitsData(),
            PublishingAccess.kWrite,
            _OpenAssetIOHost.instance().context,
        )

        # Get destination file path (potentially containing batch_num
        # placeholder). This may be a temporary/staging path, or it
        # may be the final path, depending on the manager's
        # implementation.
        file_path_tmplt = _OpenAssetIOHost.instance().resolve_to_path(
            working_ref, access_mode=ResolveAccess.kManagerDriven
        )

        results = list()
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file_path = file_path_tmplt.replace("%batch_num%", str(batch_number))

            img.save(file_path, pnginfo=metadata, compress_level=self.compress_level)

            # Configure traits to register with the asset manager.
            url = _OpenAssetIOHost.instance().file_url_path_converter.pathToUrl(file_path)
            spec.locatableContentTrait().setLocation(url)

            # Publish the image to the working reference.
            final_ref = _OpenAssetIOHost.instance().manager.register(
                working_ref,
                spec.traitsData(),
                PublishingAccess.kWrite,
                _OpenAssetIOHost.instance().context,
            )
            # Get the path of the published image. This may be different
            # to the path resolved from the working reference (i.e. if
            # the manager moved it as part of the publishing process).
            final_file_path = pathlib.Path(_OpenAssetIOHost.instance().resolve_to_path(final_ref))
            # Copy to ComfyUI temp directory for display in the UI. For
            # security reasons, ComfyUI does not allow images to be
            # served from arbitrary paths on disk, so we must copy them
            # to an allowed location. Here, we choose ComfyUI's temp
            # directory.
            shutil.copy2(final_file_path, folder_paths.get_temp_directory())

            results.append({"filename": final_file_path.name, "subfolder": "", "type": "temp"})

        # For output nodes, we can return a dict with a "ui" key,
        # containing data to display in the ComfyUI interface.
        # Here, we return the list of published images, which will be
        # displayed in the node.
        return {"ui": {"images": results}}


# Plugin registration: node classes.
NODE_CLASS_MAPPINGS = {
    "OpenAssetIOResolveImage": ResolveImage,
    "OpenAssetIOPublishImage": PublishImage,
}

# Plugin registration: node names.
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAssetIOResolveImage": "OpenAssetIO Resolve Image",
    "OpenAssetIOPublishImage": "OpenAssetIO Publish Image",
}
