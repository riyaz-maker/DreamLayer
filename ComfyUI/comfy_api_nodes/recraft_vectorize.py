from __future__ import annotations
import hashlib
import os
import torch
import numpy as np
from PIL import Image
from comfy_api_nodes.nodes_recraft import handle_recraft_file_request
from inspect import cleandoc
from typing import Optional
from comfy.utils import ProgressBar
from comfy_extras.nodes_images import SVG
from comfy.comfy_types.node_typing import IO
from comfy_api_nodes.apis.recraft_api import (
    RecraftImageGenerationRequest,
    RecraftImageGenerationResponse,
    RecraftImageSize,
    RecraftModel,
    RecraftStyle,
    RecraftStyleV3,
    RecraftColor,
    RecraftColorChain,
    RecraftControls,
    RecraftIO,
    get_v3_substyles,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    EmptyRequest,
)
from comfy_api_nodes.apinode_utils import (
    bytesio_to_image_tensor,
    download_url_to_bytesio,
    tensor_to_bytesio,
    resize_mask_to_image,
    validate_string,
)
from server import PromptServer
from io import BytesIO
from PIL import UnidentifiedImageError

class RecraftVectorizeNode:
    """
    Vectorizes a raster image using the Recraft API and saves it as a pure-vector SVG file.
    """
    RETURN_TYPES = ("STRING",)
    DESCRIPTION = cleandoc(__doc__ or "") if 'cleandoc' in globals() else __doc__
    RETURN_NAMES = ("svg_file_path",)
    FUNCTION = "execute"
    CATEGORY = "api node/image/Recraft"
    API_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IO.IMAGE,),
                "resolution_limit": (
                    IO.INT, {
                        "default": 2048, "min": 128, "max": 4096, "step": 64,
                    }
                ),
                 "output_path": (
                    IO.STRING, {
                        "default": "output/vectorized_svgs",
                        "tooltip": "Directory where the SVG files will be saved.",
                    }
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
            },
        }

    def execute(self, image: torch.Tensor, resolution_limit: int, output_path: str, **kwargs) -> tuple[str]:
        if image.shape[0] > 1:
            print("Node handles images one by one. Processing the first image.")
            image = image[0].unsqueeze(0)
        pil_image = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        if pil_image.width > resolution_limit or pil_image.height > resolution_limit:
            print(f"Image resolution ({pil_image.width}x{pil_image.height}) exceeds the recommended limit of {resolution_limit}px. This may result in longer processing times.")

        # API call to recraft
        print("Contacting Recraft API to vectorize image")
        try:
            # reusing handle_recraft_file_request function to send the image
            svg_bytes_list = handle_recraft_file_request(
                image=image,
                path="/proxy/recraft/images/vectorize",
                auth_kwargs=kwargs,
            )
            if not svg_bytes_list:
                raise ConnectionError("The Recraft API did not return any data.")

            svg_content = svg_bytes_list[0].getvalue().decode('utf-8')
            print("Successfully received SVG data from API.")

        except Exception as e:
            print(f"Recraft API Error: {e}")
            raise Exception(f"Failed to vectorize image. Reason: {e}") from e

        # save SVG content to a file
        image_hash = hashlib.sha256(pil_image.tobytes()).hexdigest()
        filename = f"vectorized_{image_hash[:16]}.svg"

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        svg_file_path = os.path.join(output_path, filename)

        with open(svg_file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)

        print(f"SVG saved to: {svg_file_path}")
        return (svg_file_path,)
    

NODE_CLASS_MAPPINGS = {
    "RecraftVectorizeNode": RecraftVectorizeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RecraftVectorizeNode": "Recraft Vectorize Image to File",
}