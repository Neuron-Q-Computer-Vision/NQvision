import unittest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
from NQvision.events import EventStreamNQvisionCore, EventStreamConfig, DetectionEvent
from NQvision.core import ModelConfig, TrackedObject


class TestEventStreamNQvisionCore(unittest.TestCase):
    def setUp(self):
        self.model_path = "dummy_model_path.onnx"
        with patch("events.NQvisionCore.__init__") as mock_init:
            mock_init.return_value = None
            self.core = EventStreamNQvisionCore(self.model_path)
            self.core.event_callbacks = []
            self.core.last_emitted = {}
            self.core.detection_frequency = {}
            self.core.logger = Mock()

    def test_event_stream_config_validation(self):
        # Test valid configuration
        config = EventStreamConfig(
            confidence_threshold=0.8,
            frequency_threshold=5,
            frequency_time_window=timedelta(seconds=10),
            class_specific_thresholds={1: 0.9, 2: 0.7},
        )
        self.assertEqual(config.confidence_threshold, 0.8)

        # Test invalid confidence threshold
        with self.assertRaises(ValueError):
            EventStreamConfig(confidence_threshold=1.5)

        # Test invalid frequency threshold
        with self.assertRaises(ValueError):
            EventStreamConfig(frequency_threshold=0)

        # Test invalid time window
        with self.assertRaises(ValueError):
            EventStreamConfig(frequency_time_window=timedelta(seconds=-1))

        # Test invalid class-specific threshold
        with self.assertRaises(ValueError):
            EventStreamConfig(class_specific_thresholds={1: 1.5})

    @patch("events.datetime")
    def test_should_emit_event(self, mock_datetime):
        current_time = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = current_time

        # Test new track
        obj = TrackedObject(track_id=1, bbox=(0, 0, 10, 10), class_id=0, confidence=0.9)
        should_emit, event_type = self.core._should_emit_event(obj)
        self.assertTrue(should_emit)
        self.assertEqual(event_type, "new_track")

        # Test rate limiting
        self.core.last_emitted[1] = current_time
        mock_datetime.now.return_value = current_time + timedelta(milliseconds=500)
        should_emit, event_type = self.core._should_emit_event(obj)
        self.assertFalse(should_emit)
        self.assertIsNone(event_type)

        # Test high confidence
        mock_datetime.now.return_value = current_time + timedelta(seconds=2)
        obj.confidence = 0.9
        should_emit, event_type = self.core._should_emit_event(obj)
        self.assertTrue(should_emit)
        self.assertEqual(event_type, "high_confidence")

    def test_create_event(self):
        obj = TrackedObject(track_id=1, bbox=(0, 0, 10, 10), class_id=0, confidence=0.9)
        event = self.core._create_event(obj, "test_event")

        self.assertEqual(event.track_id, 1)
        self.assertEqual(event.bbox, (0, 0, 10, 10))
        self.assertEqual(event.class_id, 0)
        self.assertEqual(event.confidence, 0.9)
        self.assertEqual(event.event_type, "test_event")

    @patch("events.datetime")
    async def test_emit_event(self, mock_datetime):
        current_time = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = current_time

        callback1 = AsyncMock()
        callback2 = Mock()
        await self.core.add_event_callback(callback1)
        await self.core.add_event_callback(callback2)

        event = DetectionEvent(
            timestamp=current_time.isoformat(),
            track_id=1,
            class_id=0,
            confidence=0.9,
            bbox=(0, 0, 10, 10),
            event_type="test_event",
        )

        await self.core._emit_event(event)

        callback1.assert_called_once_with(event)
        callback2.assert_called_once_with(event)
        self.assertEqual(self.core.last_emitted[event.track_id], current_time)

    @patch.object(EventStreamNQvisionCore, "process_frame_async")
    async def test_process_frame_and_emit_events(self, mock_process_frame):
        tracked_obj = TrackedObject(
            track_id=1, bbox=(0, 0, 10, 10), class_id=0, confidence=0.9
        )
        mock_process_frame.return_value = [tracked_obj]

        callback = AsyncMock()
        await self.core.add_event_callback(callback)

        events = await self.core.process_frame_and_emit_events(np.zeros((100, 100, 3)))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].track_id, 1)
        callback.assert_called_once()

    def test_configure_event_stream(self):
        self.core.configure_event_stream(
            confidence_threshold=0.7,
            frequency_threshold=3,
            frequency_time_window=5,
            emit_all_detections=True,
            class_specific_thresholds={1: 0.8},
        )

        self.assertEqual(self.core.event_stream_config.confidence_threshold, 0.7)
        self.assertEqual(self.core.event_stream_config.frequency_threshold, 3)
        self.assertEqual(
            self.core.event_stream_config.frequency_time_window, timedelta(seconds=5)
        )
        self.assertTrue(self.core.event_stream_config.emit_all_detections)
        self.assertEqual(
            self.core.event_stream_config.class_specific_thresholds, {1: 0.8}
        )


def run_async_test(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


if __name__ == "__main__":
    unittest.main()
