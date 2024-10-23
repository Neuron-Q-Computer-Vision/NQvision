from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import asyncio
from datetime import datetime, timedelta
import logging

import numpy as np
from argus.core import ARGUSCore, ModelConfig, TrackedObject
from PIL import Image


@dataclass
class DetectionEvent:
    timestamp: str
    track_id: int
    class_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]
    event_type: str  # e.g., 'new_track', 'high_confidence', 'frequent_detection'


@dataclass
class EventStreamConfig:
    confidence_threshold: float = 0.8
    frequency_threshold: int = 5  # Number of detections within time window
    frequency_time_window: timedelta = timedelta(seconds=10)
    emit_all_detections: bool = False
    class_specific_thresholds: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.frequency_threshold < 1:
            raise ValueError("frequency_threshold must be at least 1")
        if self.frequency_time_window <= timedelta(0):
            raise ValueError("frequency_time_window must be positive")
        for class_id, threshold in self.class_specific_thresholds.items():
            if not 0 <= threshold <= 1:
                raise ValueError(
                    f"Class-specific threshold for class {class_id} must be between 0 and 1"
                )


class EventStreamARGUSCore(ARGUSCore):
    def __init__(self, model_path: str, config: ModelConfig = ModelConfig()):
        super().__init__(model_path, config)
        self.event_stream_config = EventStreamConfig()
        self.detection_frequency: Dict[int, List[datetime]] = {}
        self.event_callbacks: List[Callable[[DetectionEvent], None]] = []
        self.last_emitted: Dict[int, datetime] = {}
        self.logger = logging.getLogger(__name__)

        # Create an event loop for async operations
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

    async def add_event_callback(self, callback: Callable[[DetectionEvent], None]):
        """Add a callback function to be called when events are emitted."""
        self.event_callbacks.append(callback)

    def _should_emit_event(self, obj: TrackedObject) -> Tuple[bool, Optional[str]]:
        """Determine if an event should be emitted for this detection."""
        current_time = datetime.now()
        track_id = obj.track_id

        # Rate limiting: Avoid emitting the same event repeatedly within a short period
        min_event_interval = timedelta(seconds=1)
        if (
            track_id in self.last_emitted
            and current_time - self.last_emitted[track_id] < min_event_interval
        ):
            self.logger.debug(f"Track {track_id} suppressed due to rate limiting")
            return False, None

        # Always emit for new tracks
        if track_id not in self.last_emitted:
            self.last_emitted[track_id] = (
                current_time  # Store the time of the first emission
            )
            return True, "new_track"

        # Ensure the same track_id is only emitted once within the time window
        time_window_start = (
            current_time - self.event_stream_config.frequency_time_window
        )
        if self.last_emitted[track_id] > time_window_start:
            self.logger.debug(f"Track {track_id} suppressed, within time window")
            return False, None

        # Check if we should emit based on all detections
        if self.event_stream_config.emit_all_detections:
            self.last_emitted[track_id] = current_time
            return True, "all_detections"

        # Check confidence threshold (class-specific or general)
        confidence_threshold = self.event_stream_config.class_specific_thresholds.get(
            obj.class_id, self.event_stream_config.confidence_threshold
        )
        if obj.confidence >= confidence_threshold:
            self.last_emitted[track_id] = current_time
            return True, "high_confidence"

        # Check frequency
        if track_id not in self.detection_frequency:
            self.detection_frequency[track_id] = []
        self.detection_frequency[track_id].append(current_time)

        # Remove old detections outside the time window
        self.detection_frequency[track_id] = [
            t for t in self.detection_frequency[track_id] if t > time_window_start
        ]

        if (
            len(self.detection_frequency[track_id])
            >= self.event_stream_config.frequency_threshold
        ):
            self.last_emitted[track_id] = current_time
            return True, "frequent_detection"

        self.logger.debug(f"Track {track_id} suppressed, confidence: {obj.confidence}")
        return False, None

    def _create_event(self, obj: TrackedObject, event_type: str) -> DetectionEvent:
        """Create a DetectionEvent object from a TrackedObject."""
        return DetectionEvent(
            timestamp=datetime.now().isoformat(),
            track_id=obj.track_id,
            class_id=obj.class_id,
            confidence=obj.confidence,
            bbox=obj.bbox,
            event_type=event_type,
        )

    async def _emit_event(self, event: DetectionEvent):
        """Emit an event to all registered callbacks."""
        self.last_emitted[event.track_id] = datetime.now()

        for callback in self.event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                self.logger.error(f"Error in event callback: {e}")

    async def process_frame_and_emit_events(
        self,
        frame: Union[str, Path, Image.Image, np.ndarray],
    ) -> List[DetectionEvent]:
        """
        Process a frame and emit events for detections that meet the criteria.

        Args:
            frame: Input frame

        Returns:
            List of DetectionEvent objects that were emitted
        """
        tracked_objects = await self.process_frame_async(frame)
        emitted_events = []

        for obj in tracked_objects:
            should_emit, event_type = self._should_emit_event(obj)

            if should_emit:
                event = self._create_event(obj, event_type)
                await self._emit_event(event)
                emitted_events.append(event)

        return emitted_events

    def configure_event_stream(
        self,
        confidence_threshold: Optional[float] = None,
        frequency_threshold: Optional[int] = None,
        frequency_time_window: Optional[int] = None,
        emit_all_detections: Optional[bool] = None,
        class_specific_thresholds: Optional[Dict[int, float]] = None,
    ):
        """Configure the event stream parameters."""
        if confidence_threshold is not None:
            self.event_stream_config.confidence_threshold = confidence_threshold
        if frequency_threshold is not None:
            self.event_stream_config.frequency_threshold = frequency_threshold
        if frequency_time_window is not None:
            self.event_stream_config.frequency_time_window = timedelta(
                seconds=frequency_time_window
            )
        if emit_all_detections is not None:
            self.event_stream_config.emit_all_detections = emit_all_detections
        if class_specific_thresholds is not None:
            self.event_stream_config.class_specific_thresholds.update(
                class_specific_thresholds
            )

        self.event_stream_config.validate()
