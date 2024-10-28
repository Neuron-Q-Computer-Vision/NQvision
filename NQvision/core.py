import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Union
import numpy as np
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import logging
from datetime import datetime
import onnxruntime as ort
import os
from PIL import Image
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    bbox: Tuple[int, int, int, int]
    class_id: int
    confidence: float


@dataclass
class TrackerConfig:
    max_age: int = 10
    nn_budget: Optional[int] = None
    nms_max_overlap: float = 0.4
    max_iou_distance: float = 0.3
    confidence_threshold: float = 0.5
    high_confidence_threshold: float = 0.8
    min_confidance: float = 0.69


@dataclass
class ModelConfig:
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.5
    provider: str = "CUDAExecutionProvider"
    backup_provider: str = "CPUExecutionProvider"
    tracker_config: TrackerConfig = field(default_factory=TrackerConfig)
    enable_tracker: bool = False


@dataclass
class TrackedObject:
    track_id: int
    bbox: Tuple[int, int, int, int]
    class_id: int
    confidence: float


class NQvisionCore:
    def __init__(self, model_path: str, config: ModelConfig = ModelConfig()):
        self.config = config
        self.model = self._load_model(model_path)
        self.tracker = DeepSort(
            max_age=config.tracker_config.max_age,
            nn_budget=config.tracker_config.nn_budget,
            nms_max_overlap=config.tracker_config.nms_max_overlap,
            max_iou_distance=config.tracker_config.max_iou_distance,
        )
        self.track_confidences: Dict[int, float] = {}
        self.persistent_tracks: Set[int] = set()
        self.logger = logging.getLogger(__name__)
        self.original_size: Tuple[int, int]
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _load_model(self, model_path: str) -> ort.InferenceSession:
        """Load the ONNX model with fallback to CPU if CUDA is unavailable."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            return ort.InferenceSession(model_path, providers=[self.config.provider])
        except Exception as e:
            logger.warning(
                f"Failed to load model with {self.config.provider}, "
                f"falling back to {self.config.backup_provider}: {e}"
            )
            return ort.InferenceSession(
                model_path, providers=[self.config.backup_provider]
            )

    def _format_detections_for_tracker(
        self, detections: List[DetectionResult]
    ) -> List[List]:
        """Format detections for the DeepSORT tracker."""
        formatted_detections = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            formatted_detections.append(
                [
                    [x1, y1, x2 - x1, y2 - y1],  # Convert to [x, y, width, height]
                    det.confidence,
                    det.class_id,
                ]
            )
        return formatted_detections

    def update_tracks(
        self, detections: List[DetectionResult], frame: np.ndarray
    ) -> List[TrackedObject]:
        """Update object tracks using DeepSORT."""
        formatted_detections = self._format_detections_for_tracker(detections)
        tracked_objects = self.tracker.update_tracks(formatted_detections, frame=frame)

        result_tracks = []
        for track in tracked_objects:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr()  # Convert to [x1, y1, x2, y2] format
            class_id = track.get_det_class()

            # Update confidence for this track
            if track_id not in self.track_confidences:
                self.track_confidences[track_id] = 0.0

            # Update confidence from detections
            for det in detections:
                if (
                    det.class_id == class_id
                    and det.confidence >= self.config.tracker_config.min_confidance
                ):
                    self.track_confidences[track_id] = max(
                        self.track_confidences[track_id], det.confidence
                    )

            current_confidence = self.track_confidences[track_id]

            # Check for high confidence threshold crossing
            if (
                current_confidence
                >= self.config.tracker_config.high_confidence_threshold
                and track_id not in self.persistent_tracks
            ):
                self.persistent_tracks.add(track_id)
                self._log_detection(track_id, current_confidence)

            if current_confidence >= self.config.tracker_config.min_confidance:
                result_tracks.append(
                    TrackedObject(
                        track_id=track_id,
                        bbox=tuple(map(int, bbox)),
                        class_id=class_id,
                        confidence=current_confidence,
                    )
                )

        return result_tracks

    def _log_detection(self, track_id: int, confidence: float):
        """Log high-confidence detections."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(
            f"High confidence detection - Track ID: {track_id}, "
            f"Confidence: {confidence:.2f}, Time: {current_time}"
        )

    def process_frame(
        self,
        frame: Union[str, Path, Image.Image, np.ndarray],
        return_drawn_frame: bool = False,
    ) -> Union[np.ndarray, List[TrackedObject], List[DetectionResult]]:
        """Process a frame and return tracked objects."""
        self.logger.debug("Starting frame processing")
        # If frame is a file path, open the image using PIL
        if isinstance(frame, (str, Path)):
            if not os.path.exists(frame):
                raise FileNotFoundError(f"Image file not found: {frame}")
            frame = Image.open(frame)
            frame = np.array(frame)  # Convert PIL Image to numpy array
        elif isinstance(frame, Image.Image):
            frame = np.array(frame)  # Convert PIL Image to numpy array
        elif isinstance(frame, np.ndarray):
            frame = frame.copy()  # Create a copy of the numpy array
        else:
            raise TypeError(f"Unsupported image type: {type(frame)}")
        # save original size
        self.original_size = frame.shape[:2]

        # Preprocess frame and run inference
        self.logger.debug("Preprocessing frame")
        input_data = self._preprocess_frame(frame)
        self.logger.debug("Running inference")
        output = self.run_inference(input_data)
        self.logger.debug("Processing output")
        detections = self.process_output(output)
        if self.config.enable_tracker:
            # Update tracks
            self.logger.debug("Updating tracks")
            tracked_objects = self.update_tracks(detections, frame)

            if return_drawn_frame:
                self.logger.debug("Drawing tracked objects on frame")
                return (
                    self._draw_tracked_objects(frame.copy(), tracked_objects),
                    tracked_objects,
                )
            return tracked_objects
        if return_drawn_frame:
            self.logger.debug("Drawing bounding boxes on frame")
            return self.draw_boxes(np.array(frame), detections), detections
        return detections

    async def process_frame_async(
        self,
        frame: Union[str, Path, Image.Image, np.ndarray],
        return_drawn_frame: bool = False,
    ) -> Union[np.ndarray, List[TrackedObject], List[DetectionResult]]:
        """
        Asynchronously process a frame for object detection/tracking.

        Args:
            frame: Input frame (file path, PIL Image, or numpy array)
            return_drawn_frame: If True, returns the frame with detections drawn

        Returns:
            Same as process_frame, but delivered asynchronously
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.process_frame, frame, return_drawn_frame
        )

    def _draw_tracked_objects(
        self, frame: np.ndarray, tracked_objects: List[TrackedObject]
    ) -> np.ndarray:
        """Draw tracked objects on the frame."""
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.bbox

            # Determine color based on persistence
            color = (
                (0, 0, 255) if obj.track_id in self.persistent_tracks else (0, 255, 0)
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"ID: {obj.track_id}, Class: {obj.class_id}, Conf: {obj.confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        return frame

    @staticmethod
    def draw_boxes(image: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """Draw bounding boxes and labels on the image."""
        img_copy = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img_copy,
                f"ID: {det.class_id}, Conf: {det.confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
        return img_copy

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a frame for inference.

        Args:
            frame: numpy array of the image

        Returns:
            Preprocessed image as numpy array
        """
        img_height, img_width = frame.shape[:2]
        self.original_size = (img_height, img_width)
        img = cv2.resize(frame, self.config.input_size)
        img = img / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
        return img

    def run_inference(self, input_data: np.ndarray) -> np.ndarray:
        """Run model inference on preprocessed input data."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Running inference attempt {attempt + 1}")
                return self.model.run(None, {"images": input_data})[0]
            except Exception as e:
                self.logger.error(f"Inference failed on attempt {attempt + 1}: {e}")
                if attempt + 1 == max_retries:
                    raise

    def run_batch_inference(
        self, batch_input_data: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Run inference on a batch of preprocessed input data."""
        try:
            self.logger.debug(
                f"Running batch inference on {len(batch_input_data)} frames"
            )
            return self.model.run(None, {"images": np.stack(batch_input_data)})
        except Exception as e:
            self.logger.error(f"Batch inference failed: {e}")
            raise

    def process_output(self, output: np.ndarray) -> List[DetectionResult]:
        """Process model output into a list of DetectionResult objects."""
        boxes = []
        for row in output[0]:
            prob = row[4:].max()
            if prob < self.config.confidence_threshold:
                continue

            class_id = row[4:].argmax()
            xc, yc, w, h = row[:4]
            x1 = int((xc - w / 2) * self.original_size[1])
            y1 = int((yc - h / 2) * self.original_size[0])
            x2 = int((xc + w / 2) * self.original_size[1])
            y2 = int((yc + h / 2) * self.original_size[0])

            boxes.append(
                DetectionResult(
                    bbox=(x1, y1, x2, y2),
                    class_id=int(class_id),
                    confidence=float(prob),
                )
            )

        return boxes

    async def process_batch(
        self,
        frames: List[Union[str, Path, Image.Image, np.ndarray]],
        return_drawn_frames: bool = False,
    ) -> List[Union[np.ndarray, List[TrackedObject], List[DetectionResult]]]:
        """
        Process a batch of frames asynchronously.

        Args:
            frames: List of input frames
            return_drawn_frames: If True, returns frames with detections drawn

        Returns:
            List of results, same type as process_frame for each frame
        """
        self.logger.info(f"Processing batch of {len(frames)} frames")
        preprocessed_frames = []
        for frame in frames:
            if isinstance(frame, (str, Path)):
                frame = Image.open(frame)
            if isinstance(frame, Image.Image):
                frame = np.array(frame)
            preprocessed_frames.append(self._preprocess_frame(frame))

        batch_output = self.run_batch_inference(preprocessed_frames)

        results = []
        for i, output in enumerate(batch_output):
            detections = self.process_output(output)
            if self.config.enable_tracker:
                tracked_objects = self.update_tracks(detections, frames[i])
                if return_drawn_frames:
                    results.append(
                        (
                            self._draw_tracked_objects(
                                frames[i].copy(), tracked_objects
                            ),
                            tracked_objects,
                        )
                    )
                else:
                    results.append(tracked_objects)
            else:
                if return_drawn_frames:
                    results.append((self.draw_boxes(frames[i], detections), detections))
                else:
                    results.append(detections)

        return results
