from .core import ARGUSCore, ModelConfig, TrackerConfig, DetectionResult, TrackedObject
from .events import EventStreamARGUSCore, EventStreamConfig, DetectionEvent

__all__ = [
    "ARGUSCore",
    "ModelConfig",
    "TrackerConfig",
    "DetectionResult",
    "TrackedObject",
    "EventStreamARGUSCore",
    "EventStreamConfig",
    "DetectionEvent",
]

__version__ = "0.1.0"
