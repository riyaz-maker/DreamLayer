import unittest
from unittest.mock import patch
import torch
import numpy as np
import os
import hashlib
from io import BytesIO
from comfy_api_nodes.recraft_vectorize import RecraftVectorizeNode

class TestRecraftVectorizeNode(unittest.TestCase):
    """
    Unit test for the RecraftVectorizeNode.
    """

    def setUp(self):
        """Set up the environment for each test case."""
        self.node = RecraftVectorizeNode()
        self.output_dir = "output/test_svgs"
        # Create a consistent dummy image for deterministic hashing
        dummy_array = np.zeros((64, 64, 3), dtype=np.uint8)
        dummy_array[16:48, 16:48, :] = 128
        self.image_tensor = torch.from_numpy(dummy_array.astype(np.float32) / 255.0).unsqueeze(0)
        # Define the exact SVG content our mock API will return
        self.mock_svg_content = '<svg width="64" height="64"><rect x="16" y="16" width="32" height="32" fill="#808080"/></svg>'

    def tearDown(self):
        """Clean up any files or directories created during tests."""
        if os.path.exists(self.output_dir):
            for f in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, f))
            os.rmdir(self.output_dir)

    @patch('comfy_api_nodes.recraft_vectorize.handle_recraft_file_request')
    def test_execute_saves_file_with_correct_checksum(self, mock_api_call):
        """
        GIVEN a raster image tensor
        WHEN the node's execute method is called
        THEN it should save an SVG file with the correct size and checksum.
        """
        # Setup MOCK
        mock_api_call.return_value = [BytesIO(self.mock_svg_content.encode('utf-8'))]
        result_path_str, = self.node.execute(
            image=self.image_tensor,
            resolution_limit=1024,
            output_path=self.output_dir,
            auth_token="test_token"
        )
        result_path = result_path_str.strip()

        self.assertTrue(os.path.exists(result_path))
        self.assertEqual(os.path.getsize(result_path), len(self.mock_svg_content.encode('utf-8')))

        # Verify the checksum of the ENTIRE file for accurate validation
        with open(result_path, 'rb') as f:
            checksum = hashlib.sha256(f.read()).hexdigest()
        expected_checksum = hashlib.sha256(self.mock_svg_content.encode('utf-8')).hexdigest()
        self.assertEqual(checksum, expected_checksum)