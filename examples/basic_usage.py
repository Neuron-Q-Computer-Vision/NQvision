# Example usage
import cv2
from vidgear.gears import VideoGear

from argus.core import ARGUSCore, ModelConfig, TrackerConfig
from argus.events import DetectionEvent, EventStreamARGUSCore


def VideoDetection():
    config = ModelConfig(
        tracker_config=TrackerConfig(
            max_age=10, high_confidence_threshold=0.8, min_confidance=0.2
        )
    )
    detector = ARGUSCore("./models/argus/cpu/argus.onnx", config)

    # Example of processing a single image
    image_path = "./test/wpn_onnx_test.jpg"
    # results = detector.process_frame(image_path)
    # logger.info(f"Processed single image with {len(results)} detections")

    # Example video processing
    cap = VideoGear(source="./test/toro_in.mp4").start()
    frame_count = 0
    while True:
        frame = cap.read()
        frame_count += 1
        if frame is None:
            break

        if frame_count % 5 == 0:
            frame = detector.process_frame(frame, return_drawn_frame=True)
        cv2.imshow("Tracked Objects", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


### Example
import asyncio
import cv2

# from argus_event_stream import EventStreamARGUSCore, ModelConfig, DetectionEvent


async def example_callback(event: DetectionEvent):
    print(f"Event received: {event.event_type}")
    print(
        f"Track ID: {event.track_id}, Class: {event.class_id}, Confidence: {event.confidence}"
    )


async def process_video_with_events():
    # Initialize
    model_path = "./models/argus/cpu/argus.onnx"
    config = ModelConfig(enable_tracker=True)
    detector = EventStreamARGUSCore(model_path, config)

    # Configure event stream
    detector.configure_event_stream(
        confidence_threshold=0.6,
        frequency_threshold=1,
        frequency_time_window=5,  # 5 seconds
    )

    # Add callback
    detector.add_event_callback(example_callback)

    # Process video
    cap = cv2.VideoCapture("./test/toro_in.mp4")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        events = await detector.process_frame_and_emit_events(frame)

        # Additional processing if needed
        for event in events:
            # if event.event_type == "high_confidence":
            #     print(f"High confidence detection: Class {event.class_id}")
            print(event)

    cap.release()


# Run the example
asyncio.run(process_video_with_events())
