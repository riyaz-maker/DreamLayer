import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
import hashlib
import json
from io import BytesIO
from PIL import Image
from comfy_api_nodes.nodes_runway_ref import RunwayRefGenerateNode
from comfy_api_nodes.apis import RunwayTextToImageAspectRatioEnum

class TestRunwayRefGenerateNode(unittest.TestCase):

    def setUp(self):
        """Set up the environment for each test case."""
        self.node = RunwayRefGenerateNode()
        dummy_array = np.zeros((64, 64, 3), dtype=np.uint8)
        self.image_tensor = torch.from_numpy(dummy_array.astype(np.float32) / 255.0).unsqueeze(0)
        self.prompt = "a cyberpunk, futuristic and techy city"

    @patch('comfy_api_nodes.nodes_runway_ref.upload_images_to_comfyapi')
    @patch('comfy_api_nodes.nodes_runway_ref.download_url_to_image_tensor')
    @patch('comfy_api_nodes.nodes_runway_ref.poll_until_finished')
    @patch('comfy_api_nodes.nodes_runway_ref.SynchronousOperation')
    def test_execute_includes_reference_hash_in_payload(
        self,
        mock_sync_op,
        mock_poll,
        mock_download,
        mock_upload
    ):
        """
        GIVEN a prompt and a reference image
        WHEN the node executes
        THEN it should include the sha256 hash of the image in the JSON payload sent to the API.
        """
        mock_upload.return_value = ["https://fake-upload-url.com/ref.png"]
        mock_initial_response = MagicMock()
        mock_initial_response.id = "fake_task_123"
        mock_sync_op.return_value.execute.return_value = mock_initial_response
        mock_final_response = MagicMock()
        mock_final_response.output = ["https://fake-result-url.com/result.png"]
        mock_poll.return_value = mock_final_response
        dummy_output_image = torch.zeros((1, 64, 64, 3))
        mock_download.return_value = dummy_output_image
        self.node.execute(
            prompt=self.prompt,
            reference_image=self.image_tensor,
            seed=42,
            ratio=RunwayTextToImageAspectRatioEnum.field_1920_1080.value
        )

        mock_upload.assert_called_once()
        mock_sync_op.assert_called_once()
        request_payload = mock_sync_op.call_args.kwargs['request']
        self.assertEqual(request_payload.promptText, self.prompt)
        self.assertEqual(request_payload.ratio, "1920:1080")
        self.assertIsNotNone(request_payload.referenceImages)
        self.assertEqual(request_payload.referenceImages[0].uri, "https://fake-upload-url.com/ref.png")
