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
        """
        Set up the environment for each test case.
        """
        self.node = RecraftVectorizeNode()
        self.output_dir = "output/test_svgs"
        # creating a dummy image for testing
        dummy_array = np.zeros((64, 64, 3), dtype=np.uint8)
        # gray square
        dummy_array[16:48, 16:48, :] = 128 
        self.image_tensor = torch.from_numpy(dummy_array.astype(np.float32) / 255.0).unsqueeze(0)
        self.mock_svg_content = '<svg width="64" height="64"><rect x="16" y="16" width="32" height="32" fill="#808080"/></svg>'

    def tearDown(self):
        """
        Clean up any files or directories created during tests.
        """
        if os.path.exists(self.output_dir):
            for f in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, f))
            os.rmdir(self.output_dir)

    @patch('comfy_api_nodes.recraft_vectorize.handle_recraft_file_request')
    def test_execute_saves_file_with_correct_checksum(self, mock_api_call):
        mock_api_call.return_value = [BytesIO(self.mock_svg_content.encode('utf-8'))]

        # executing the node
        result_path, = self.node.execute(
            image=self.image_tensor,
            resolution_limit=1024,
            output_path=self.output_dir,
            auth_token="test_token"
        )

        mock_api_call.assert_called_once()

        # verifications
        self.assertEqual(mock_api_call.call_args[1]['path'], "/proxy/recraft/images/vectorize")
        self.assertTrue(os.path.exists(result_path))
        self.assertEqual(os.path.getsize(result_path), len(self.mock_svg_content.encode('utf-8')))
        with open(result_path, 'rb') as f:
            checksum = hashlib.sha256(f.read(1024)).hexdigest()
        expected_checksum = hashlib.sha256(self.mock_svg_content.encode('utf-8')).hexdigest()
        self.assertEqual(checksum, expected_checksum)
