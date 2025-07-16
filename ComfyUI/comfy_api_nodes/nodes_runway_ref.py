from __future__ import annotations
from inspect import cleandoc
import hashlib
import json
import logging
import torch
from typing import Optional
from comfy.comfy_types.node_typing import IO
from comfy.utils import ProgressBar
from .apis import (
    RunwayTaskStatusResponse as TaskStatusResponse,
    RunwayTextToImageRequest,
    ReferenceImage,
    RunwayTextToImageAspectRatioEnum,
    Model4,
)
from .apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    EmptyRequest,
)
from .nodes_runway import (
    poll_until_finished,
    get_image_url_from_task_status,
    PATH_GET_TASK_STATUS,
    PATH_TEXT_TO_IMAGE,
    RunwayApiError,
    validate_input_image,
)
from .apinode_utils import (
    download_url_to_image_tensor,
    upload_images_to_comfyapi,
)
from .mapper_utils import model_field_to_node_input

logger = logging.getLogger(__name__)

class RunwayRefGenerateNode:
    """
    Generates an image from a text prompt using a reference image to steer the style.
    """
    DESCRIPTION = cleandoc(__doc__ or "")
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "api node/image/Runway"
    API_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (IO.STRING, {"multiline": True, "default": "a beautiful landscape"}),
                "reference_image": (IO.IMAGE,),
                "seed": (IO.INT, {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "ratio": model_field_to_node_input(
                    IO.COMBO,
                    RunwayTextToImageRequest,
                    "ratio",
                    enum_type=RunwayTextToImageAspectRatioEnum,
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    def execute(self, prompt: str, reference_image: torch.Tensor, seed: int, ratio: str, unique_id: Optional[str] = None, **kwargs) -> tuple[torch.Tensor]:
        all_output_images = []
        pbar = ProgressBar(reference_image.shape[0])
        for i in range(reference_image.shape[0]):
            img_tensor_single = reference_image[i].unsqueeze(0)
            pbar.update(1)
            logger.info(f"Processing image {i+1}/{reference_image.shape[0]} for Runway reference generation.")
            validate_input_image(img_tensor_single)

            try:
                download_urls = upload_images_to_comfyapi(
                    img_tensor_single, max_images=1, mime_type="image/png", auth_kwargs=kwargs
                )
                if not download_urls:
                    raise RunwayApiError("Failed to upload reference image.")
                ref_image_url = str(download_urls[0])

                request_payload = RunwayTextToImageRequest(
                    promptText=prompt,
                    seed=seed,
                    ratio=ratio,
                    model=Model4.gen4_image,
                    referenceImages=[ReferenceImage(uri=ref_image_url)]
                )
                initial_operation = SynchronousOperation(
                    endpoint=ApiEndpoint(
                        path=PATH_TEXT_TO_IMAGE,
                        method=HttpMethod.POST,
                        request_model=RunwayTextToImageRequest,
                        response_model=TaskStatusResponse,
                    ),
                    request=request_payload,
                    auth_kwargs=kwargs,
                )
                initial_response = initial_operation.execute()
                if not initial_response.id:
                    raise RunwayApiError("Invalid initial response from Runway API.")
                task_id = initial_response.id
                logger.info(f"Runway task submitted with ID: {task_id}")

                final_response = poll_until_finished(
                    auth_kwargs=kwargs,
                    api_endpoint=ApiEndpoint(
                        path=f"{PATH_GET_TASK_STATUS}/{task_id}",
                        method=HttpMethod.GET,
                        request_model=EmptyRequest,
                        response_model=TaskStatusResponse,
                    ),
                    node_id=unique_id,
                )
                logger.info(f"Runway task {task_id} succeeded.")

                image_url = get_image_url_from_task_status(final_response)
                if not image_url:
                     raise RunwayApiError("Runway task succeeded but no image URL was found.")
                output_image = download_url_to_image_tensor(image_url)
                all_output_images.append(output_image)

            except Exception as e:
                logger.error(f"Runway reference generation failed. Reason: {e}")
                raise ConnectionError(f"Runway task failed. Reason: {e}") from e

        if not all_output_images:
            raise RuntimeError("Runway generation produced no output images.")
        return (torch.cat(all_output_images, dim=0),)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "RunwayRefGenerate": RunwayRefGenerateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunwayRefGenerate": "Runway Reference Generate",
}
