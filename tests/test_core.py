import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import numpy as np
from PIL import Image

# Update this import to match your actual module name
from argus.core import ARGUSCore, ModelConfig, DetectionResult


class TestARGUSCore(unittest.TestCase):
    @patch("onnxruntime.InferenceSession")
    def setUp(self, mock_session):
        self.mock_session = mock_session
        self.config = ModelConfig()
        self.detector = ARGUSCore("./models/argus/cpu/argus.onnx", self.config)

    def test_preprocess_image(self):
        # Create a dummy image for testing
        test_image = Image.new("RGB", (100, 100), color="red")

        # Process the image
        processed, original = self.detector.preprocess_image(test_image)

        # Assert the preprocessed image has the correct shape
        self.assertEqual(processed.shape, (1, 3, 640, 640))
        self.assertIsInstance(original, Image.Image)

    # Update patch path to match your import
    @patch("nq_argus.ARGUSCore.run_inference")
    def test_process_output(self, mock_inference):
        # Mock inference output
        mock_output = np.array(
            [[[0.5, 0.5, 0.2, 0.2, 0.9, 0.1, 0.1]]]  # Example output for one detection
        )

        detections = self.detector.process_output(mock_output)

        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections[0], DetectionResult)
        self.assertEqual(len(detections[0].bbox), 4)

    def test_invalid_image_path(self):
        with self.assertRaises(FileNotFoundError):
            self.detector.process_image("./test/nonexistent_image.jpg")

    # Update patch paths to match your import
    @patch("nq_argus.ARGUSCore.run_inference")
    @patch("nq_argus.ARGUSCore.preprocess_image")
    def test_process_image(self, mock_preprocess, mock_inference):
        # Mock the preprocessing and inference
        mock_preprocess.return_value = (
            np.zeros((1, 3, 640, 640)),
            Image.new("RGB", (640, 640)),
        )
        mock_inference.return_value = np.array([[[0.5, 0.5, 0.2, 0.2, 0.9, 0.1, 0.1]]])

        results = self.detector.process_image("./test/wpn_onnx_test.jpg")

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], DetectionResult)


class TestModelConfig(unittest.TestCase):
    def test_default_config(self):
        config = ModelConfig()
        self.assertEqual(config.input_size, (640, 640))
        self.assertEqual(config.confidence_threshold, 0.5)
        self.assertEqual(config.provider, "CUDAExecutionProvider")


if __name__ == "__main__":
    unittest.main()
