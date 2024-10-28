# NQvision

NQvision is a powerful library built around Ultralytics models in ONNX format, designed to simplify the development of AI-driven object detection and tracking solutions. It transforms complex computer vision capabilities into an accessible, production-ready solution that revolutionizes how organizations approach real-time monitoring and security.

## 🚀 Features

### Core Capabilities

- **ONNX Model Integration**: Seamless integration with Ultralytics models
- **Real-Time Object Detection**: Optimized for immediate recognition and action
- **Continuous Object Tracking**: Advanced tracking maintaining object identities across frames
- **High-Performance Processing**: Efficient operation on both CPU and GPU
- **Customizable Detection Settings**: Adjustable confidence thresholds and tracking configurations
- **Scalable Architecture**: Handles multiple video feeds simultaneously

### Event Management

- **Real-Time Event Alerts**: Instant notification system for critical detections
- **Event Aggregation**: Intelligent clustering of detections to reduce false positives
- **Customizable Criteria**: Configurable detection thresholds and frequency parameters
- **High-Confidence Alerts**: Aggregated detection within defined time windows
- **Scalable Event Management**: Suitable for both small setups and enterprise deployments

## 💫 Key Benefits

### Unmatched Flexibility

- Universal Ultralytics Compatibility
- Expanding Architecture Support
- Adaptable Integration with existing security infrastructure

### Enterprise-Grade Performance

- Scalable from single-camera setups to city-wide deployments
- Resource-optimized processing
- Built for 24/7 mission-critical environments

### Revolutionary Features

- Intelligent Tracking across camera views
- Event Streaming with customizable detection criteria
- Automated Response System
- Multi-Camera Coordination
- Seamless handling of multiple video streams

## 🎯 Impact

### For Developers

- Eliminates the need to develop intricate AI pipelines from scratch
- Provides a ready-to-use framework for advanced surveillance
- Customizable settings and real-time capabilities
- Implement AI detection without deep AI expertise

### For Companies

- Accelerate deployment of AI-driven surveillance systems
- Minimize development costs
- Improve system reliability
- Handle complex, large-scale environments
- Event-driven architecture for prompt action on high-risk detections

## ⚡ Quick Start

### Dependencies

To install NQvision Dependencies, follow these steps:

- Install NQvision :

```bash
pip install NQvision
```

- install onnxruntime :

  - For cpu only inference :

  ```bash
  pip install onnxruntime
  ```

  - For gpu accelerated inference

  ```bash
  pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

  For CUDA 11.X (default):
  pip install onnxruntime-gpu
  ```

## Verifying the Installation

To verify that NQvision is installed correctly, run the following Python code:

```python
from NQvision.core import NQvisionCore, ModelConfig

# Create a basic configuration
config = ModelConfig(input_size=(640, 640), confidence_threshold=0.4)

# Initialize NQvisionCore (replace with your model path)
detector = NQvisionCore("path/to/model/model.onnx", config)

print("NQvision initialized successfully!")
```

If you see the success message without any errors, NQvision is installed and ready to use.

# Usage

## Table of Contents

1. Introduction
2. NQvisionCore Class
   - Configuration
   - Basic Usage
   - Advanced Usage
3. EventStreamNQvisionCore Class
   - Configuration
   - Usage
4. Examples

# Introduction

NQvision is a powerful object detection and tracking system. We provide two main classes:

1. `NQvisionCore`: For basic object detection and tracking.
2. `EventStreamNQvisionCore`: Extends `NQvisionCore` with event streaming capabilities.

This guide will help you understand how to use these classes effectively for various scenarios.

## NQvisionCore Class

The `NQvisionCore` class is the foundation of the NQvision system, providing object detection and tracking capabilities.

### Configuration

The `NQvisionCore` class is configured using the `ModelConfig` dataclass:

```python
@dataclass
class ModelConfig:
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.5
    provider: str = "CUDAExecutionProvider"
    backup_provider: str = "CPUExecutionProvider"
    tracker_config: TrackerConfig = field(default_factory=TrackerConfig)
    enable_tracker: bool = False

```

Let's break down each parameter:

- `input_size`: The input size for the model. Default is (640, 640).
- `confidence_threshold`: Minimum confidence score for detections. Default is 0.5. Adjust to balance between false positives and false negatives.
- `provider`: Primary execution provider for ONNX Runtime. Default is CUDA for GPU acceleration.
- `backup_provider`: Fallback execution provider. Default is CPU.
- `tracker_config`: Configuration for the DeepSORT tracker (explained below).
- `enable_tracker`: Whether to enable object tracking. Default is False.

<aside>

> Why `enable_tracker` must be omitted for single-frame inference:

When processing individual frames, the concept of object tracking is unnecessary because there is no temporal relationship between frames. Object tracking is designed to maintain the identity of objects across consecutive frames in a video stream. In single-frame inference, you're only interested in detecting objects in the current frame, so enabling the tracker would waste resources and add complexity without any benefit.

</aside>

<aside>

> Importance of `enable_tracker` in video/stream inference:

When working with video or streams, enabling the tracker is essential for tracking objects across multiple frames. It ensures that objects maintain consistent IDs over time, which is crucial for scenarios like monitoring moving objects (e.g., people or vehicles) in surveillance systems. Without tracking, the system would treat the same object in different frames as new detections, which could lead to fragmented and unreliable results.

</aside>

The `process_frame` method in the `NQvisionCore` class is the primary entry point for running inference on a given image or frame. It takes two parameters:

- **`frame`**: This is the actual image or video frame to be processed by the model. The model performs inference on this frame to detect objects.
- **`return_drawn_frame`**: A boolean flag that determines the format of the method's return value. If `return_drawn_frame` is set to `False`, the method will return only the predictions in the form of a `DetectionResult`object. This object contains:
  - `bbox`: A tuple of four integers representing the bounding box of the detected object (x1, y1, x2, y2).
  - `class_id`: The ID of the detected object class.
  - `confidence`: The confidence score of the detection.
    If `return_drawn_frame` is set to `True`, the method will return both the predictions and the original frame with the detected bounding boxes drawn on top. This is useful for visualising the detections.
    The second parameter, `return_drawn_frame`, allows flexibility in the output depending on the use case. If you only need the raw prediction data, you can set it to `False` to avoid unnecessary drawing operations, which can save processing time. On the other hand, setting it to `True` provides a complete visualization of the detections.
  ***

The `TrackerConfig` dataclass configures the DeepSORT tracker:

```python
@dataclass
class TrackerConfig:
    max_age: int = 10
    nn_budget: Optional[int] = None
    nms_max_overlap: float = 0.4
    max_iou_distance: float = 0.3
    confidence_threshold: float = 0.5
    high_confidence_threshold: float = 0.8
    min_confidance: float = 0.69

```

These parameters fine-tune the tracking behavior:

- `max_age`: Maximum number of frames to keep lost tracks.
- `nn_budget`: Maximum size of the appearance descriptor gallery.
- `nms_max_overlap`: Non-maximum suppression threshold.
- `max_iou_distance`: Maximum IOU distance for match candidates.
- `confidence_threshold`: Minimum confidence for a detection to be considered.
- `high_confidence_threshold`: Threshold for high-confidence detections.
- `min_confidance`: Minimum confidence for a track to be maintained.

### Basic Usage

Here's a simple example of how to use `NQvisionCore`:

```python
from NQvision.core import NQvisionCore, ModelConfig

# Create a basic configuration
config = ModelConfig(
    input_size=(640, 640),  # Adjust based on your model's input size
    confidence_threshold=0.4,  # Lower threshold for more detections, adjust as needed
    enable_tracker=True  # Enable tracking if you need to follow objects across frames
)

# Initialize NQvisionCore
detector = NQvisionCore("path/to/your/model.onnx", config)

# Process a single frame
frame = cv2.imread("path/to/your/image.jpg")
drawn_frame, detections = detector.process_frame(frame, return_drawn_frame=True)

# Display the result
cv2.imshow("Detections", drawn_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print detection results
for det in detections:
    print(f"Class: {det.class_id}, Confidence: {det.confidence}, BBox: {det.bbox}")

```

### Advanced Usage

For more advanced scenarios, such as video processing or real-time streams, you can use `NQvisionCore` in a loop:

```python
from NQvision.core import NQvisionCore, ModelConfig
from vidgear.gears import VideoGear

config = ModelConfig(
    input_size=(640, 640),
    confidence_threshold=0.5,
    enable_tracker=True,
    tracker_config=TrackerConfig(
        max_age=30,  # Keep lost tracks for longer, useful for occlusions
        max_iou_distance=0.7,  # More lenient IOU matching for faster objects
        min_confidance=0.3  # Lower confidence threshold for maintained tracks
    )
)

detector = NQvisionCore("path/to/your/model.onnx", config)

# Open video stream (can be a file or RTSP stream)
stream = VideoGear(source="path/to/your/video.mp4").start()

while True:
    frame = stream.read()
    if frame is None:
        break

    drawn_frame, tracked_objects = detector.process_frame(frame, return_drawn_frame=True)

    # Display the result
    cv2.imshow("Video Stream", drawn_frame)

    # Process tracked objects (e.g., count, analyze trajectories)
    for obj in tracked_objects:
        print(f"Track ID: {obj.track_id}, Class: {obj.class_id}, Confidence: {obj.confidence}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()

```

## EventStreamNQvisionCore Class

The `EventStreamNQvisionCore` class is an extension of `NQvisionCore` designed to handle event streaming based on object detections. While `NQvisionCore` focuses on running model inference and tracking, `EventStreamNQvisionCore` adds functionality to emit events when certain detection criteria are met. This is critical for real-time systems that need to respond to detected objects, such as triggering alerts as soon as a target is detected .

### EventStreamNQvisionCore Configuration

In addition to the `ModelConfig`, the `EventStreamNQvisionCore` class uses an `EventStreamConfig`:

```python
@dataclass
class EventStreamConfig:
    confidence_threshold: float = 0.8
    frequency_threshold: int = 5
    frequency_time_window: timedelta = timedelta(seconds=10)
    emit_all_detections: bool = False
    class_specific_thresholds: Dict[int, float] = field(default_factory=dict)

```

> In the context of real-time object detection, the importance of the `EventStreamNQvisionCore` class lies in its ability to provide structured, real-time alerts based on detected events. While the base `NQvisionCore` class focuses on detecting objects in individual frames, the extended `EventStreamNQvisionCore` class allows the system to intelligently aggregate and interpret those detections, emitting events when certain conditions are met. This allows for more sophisticated real-time monitoring and alerting, ensuring the system is not overwhelmed by individual frame detections but instead focuses on patterns and high-confidence events.

The configuration of the `EventStreamNQvisionCore` class is defined by the `EventStreamConfig` dataclass, which introduces several parameters to control when and how events are emitted:

- **`confidence_threshold`**: This parameter sets the minimum confidence required for an event to be considered high-confidence. For instance, in a particular detection use case, we might only want to trigger an alert if an object is detected with a confidence above `0.8` to avoid false positives.
- **`frequency_threshold`**: Defines how many detections of a particular class are required within a given time window for a frequent detection event. This is particularly useful for identifying persistent events. For example, if an object is detected five times within a 10-second window, an alert can be raised, indicating consistent presence in the scene.
- **`frequency_time_window`**: The time window during which the system tracks how often an object is detected. In real-time detection, you might want to ensure that multiple detections within a short period trigger an event, as this indicates the object remains visible and is not a one-off detection.
- **`emit_all_detections`**: A boolean flag that controls whether every detection is emitted as an event, or only those that meet the confidence and frequency criteria. In most cases, for a real-time detection system, you would set this to `False` to avoid overwhelming the system with low-confidence or isolated detections. Instead, you would focus on emitting significant events.
- **`class_specific_thresholds`**: This dictionary enables you to define different confidence thresholds for specific object classes. For instance, in a real-time detection system, you might set a higher confidence threshold for certain high-priority objects and a lower threshold for others, depending on the level of importance or potential risk associated with each class.

<aside>

In summary, `EventStreamNQvisionCore` transforms raw detections into actionable events, ensuring that the detection system is both reliable and responsive to real-world events.

</aside>

### EventStreamNQvisionCore Usage

Here's an example of how to use `EventStreamNQvisionCore`:

```python
import asyncio
from NQvision.events import EventStreamNQvisionCore, ModelConfig, DetectionEvent
from vidgear.gears import VideoGear

# the function to be executed when an event is detected
async def event_callback(event: DetectionEvent):
    print(f"Event: {event.event_type}, Class: {event.class_id}, Confidence: {event.confidence}")

async def process_video():
    config = ModelConfig(enable_tracker=True)
    detector = EventStreamNQvisionCore("path/to/your/model.onnx", config)

    detector.configure_event_stream(
        confidence_threshold=0.6,  # Adjust based on your needs
        frequency_threshold=3,     # Emit event after 3 detections
        frequency_time_window=5    # Within 5 seconds
    )

    # Register the callback function
    # PS: you can register multiple functions
    detector.add_event_callback(event_callback)

    camera = VideoGear(source="rtsp://username:password@ip_address:port/stream_path").start()

    while True:
        frame = camera.read()
        if frame is None:
            break

        drawn_frame, events = await detector.process_frame_and_emit_events(frame)

        cv2.imshow("Video", drawn_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.stop()
    cv2.destroyAllWindows()

asyncio.run(process_video())

```

### Examples

### Video Inference with Tracking

Here's an example of how to perform video inference with tracking enabled, utilizing the `return_drawn_frame`parameter:

```python
from NQvision.core import NQvisionCore, ModelConfig, TrackerConfig
from vidgear.gears import VideoGear
import cv2

# Set up configuration as shown above# ...# Initialize NQvisionCore
detector = NQvisionCore("path/to/model.onnx", config=model_config)

# Open video stream
video = VideoGear(source="path/to/video.mp4").start()

while True:
    frame = video.read()
    if frame is None:
        break

# Process frame with tracking and get the drawn frame
    tracked_objects, drawn_frame = detector.process_frame(frame, return_drawn_frame=True)

# Display result
    cv2.imshow("Video Tracking", drawn_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.stop()
cv2.destroyAllWindows()

```

### RTSP Stream Inference with Tracking

Similarly, for RTSP stream inference with tracking:

```python
from NQvision.core import NQvisionCore, ModelConfig, TrackerConfig
from vidgear.gears import VideoGear
import cv2

# Set up configuration as shown above# ...# Initialize NQvisionCore
detector = NQvisionCore("path/to/model.onnx", config=model_config)

# Open RTSP stream
stream = VideoGear(source="rtsp://username:password@ip_address:port/stream_path").start()

while True:
    frame = stream.read()
    if frame is None:
        break

# Process frame with tracking and get the drawn frame
    tracked_objects, drawn_frame = detector.process_frame(frame, return_drawn_frame=True)

# Display result
    cv2.imshow("RTSP Stream Tracking", drawn_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()

```

By using `return_drawn_frame=True`, you can get the frame with detections or tracked objects already drawn, simplifying the visualization process. This is especially useful when you want to quickly display or save the results without implementing custom drawing logic.

## 🔄 Current Support

- Currently supporting models such as rtlder
- Designed for future expansion
- Regular updates and expanding capabilities

## 🛠 Integration

### Deployment Features

- Rapid deployment: Operational in minutes
- Immediate enhancement of surveillance capabilities
- Minimal training requirements
- Intuitive system for security teams

### System Requirements

- Compatible with existing cameras and systems
- Supports both CPU and GPU processing
- Scalable for various deployment sizes

## 🔮 Future Development

NQvision is designed for continuous evolution, with plans to:

- Adopt additional models and architectures
- Expand ecosystem support
- Regular feature updates
- Enhanced capabilities based on community feedback

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for details. By using this library, you agree to comply with the terms and conditions outlined in this license.

## 🤝 Contributing

We welcome contributions from developers and enthusiasts eager to enhance and expand NQVision. If you’re interested in contributing, please follow these steps:

Fork the Repository: Make a personal copy of this repository by forking it.
Create a Branch: Use a descriptive name for your branch (e.g., feature/add-new-model-support).
Make Your Changes: Implement your feature or fix, ensuring to follow our code style guidelines.
Submit a Pull Request: Once your changes are ready, submit a pull request to the main branch, and provide a concise description of the modifications.
For major changes, please open an issue first to discuss your proposed additions or modifications with the team.

## 📞 Support

[\[Support\]](https://www.linkedin.com/company/neuron-q/)

---

Developed by Neuron Q | Making advanced surveillance technology accessible
