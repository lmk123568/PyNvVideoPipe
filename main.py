import os
import time

import supervision as sv
import torch


class VideoPipe(torch.multiprocessing.Process):
    def __init__(self, gpu, input_url, output_url):
        super().__init__()
        self.gpu = gpu
        self.input_url = input_url
        self.output_url = output_url

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    def run(self):
        # Import hardware-accelerated codec and YOLO detection model
        import nv_accel

        # Initialize CUDA-based video decoder with desired resolution and reconnection settings
        decoder = nv_accel.Decoder(
            self.input_url,
            enable_frame_skip=False,  # Ensure every frame is decoded
            output_width=1024,
            output_height=576,
            enable_auto_reconnect=True,  # Auto-reconnect on network interruption
            reconnect_delay_ms=10000,  # Wait 10s before each reconnection attempt
            max_reconnects=0,  # Unlimited reconnection attempts
            open_timeout_ms=5000,  # Timeout for opening the stream
            read_timeout_ms=5000,  # Timeout for reading packets
            buffer_size=4 * 1024 * 1024,  # 4MB buffer for jitter tolerance
            max_delay_ms=200,  # Max allowed decoding delay
            reorder_queue_size=32,  # B-frame reorder queue length
            decoder_threads=2,  # Number of decoder threads
            surfaces=2,  # Number of CUDA surfaces for buffering
            hwaccel="cuda",  # Use NVIDIA hardware acceleration
        )

        # Initialize CUDA-based H.264 encoder matching decoder's resolution
        # encoder = nv_accel.Encoder(
        #     output_url=self.output_url,
        #     width=decoder.get_width(),
        #     height=decoder.get_height(),
        #     fps=25.0,
        #     codec="h264",
        #     bitrate=1_000_000,  # 1 Mbps target bitrate
        # )

        # Load TensorRT-optimized YOLOv26 nano model for object detection
        yolo = nv_accel.Yolo26DetTRT(
            engine_path="./yolo26n_1x3x576x1024_fp16.engine",
            device_id=0,
            conf_thres=0.25,  # Confidence threshold for detections
        )

        # Supervision annotators and tracker for visualization
        # box_annotator = sv.BoxAnnotator()
        # label_annotator = sv.LabelAnnotator()
        # tracker = sv.ByteTrack()
        # trace_annotator = sv.TraceAnnotator()

        # Frame counter for profiling
        frame_count = 0

        # timing accumulators for profiling each pipeline stage
        sum_wait = 0
        sum_det = 0
        sum_track = 0
        sum_draw = 0
        sum_encode = 0
        sum_event = 0

        while 1:
            t0 = time.time()  # Start timing for frame-wait stage

            # Fetch next decoded frame; pts is presentation timestamp
            try:
                frame, pts = decoder.next_frame()
            except Exception as e:
                print(f"Error: {e}")
                continue

            frame_count += 1
            t1 = time.time()  # End of wait stage, start of detection

            # Run YOLO inference on GPU tensor
            det_results = yolo(frame)

            # Additional models can be added for secondary inference
            # Ensure every model's inputs and outputs remain GPU tensors
            # Avoid CPU <-> GPU transfers during model inference
            # eg:
            # cls_results = cls(frame)
            # seg_results = seg(frame)

            t2 = time.time()  # End of detection stage

            # # Convert GPU tensor to numpy and build supervision.Detections
            # det_results = det_results.cpu().numpy()
            # det_results = sv.Detections(
            #     xyxy=det_results[:, :4],
            #     confidence=det_results[:, 4],
            #     class_id=det_results[:, 5].astype(int),
            # )
            # # Update tracker with current detections
            # tracker_results = tracker.update_with_detections(det_results)
            t3 = time.time()  # End of tracking stage

            # # Move frame back to CPU for annotation
            # annotated_frame = frame.cpu().numpy()

            # # Build label text: tracker_id + class_id
            # labels = [
            #     f"#{tracker_id} {class_id}"
            #     for tracker_id, class_id in zip(
            #         tracker_results.tracker_id, tracker_results.class_id
            #     )
            # ]

            # # Draw boxes, traces and labels on the frame
            # annotated_frame = box_annotator.annotate(
            #     scene=annotated_frame, detections=tracker_results
            # )
            # annotated_frame = trace_annotator.annotate(
            #     scene=annotated_frame, detections=tracker_results
            # )
            # annotated_frame = label_annotator.annotate(
            #     scene=annotated_frame, detections=tracker_results, labels=labels
            # )
            t4 = time.time()  # End of drawing stage

            # # Send annotated frame back to GPU and encode
            # annotated_frame = torch.from_numpy(annotated_frame).to("cuda")
            # encoder.encode(annotated_frame, pts)
            t5 = time.time()  # End of encode stage

            # Business Logic
            # is_person = event(tracker_results)
            # if is_person:
            #     print("Person detected!")
            t6 = time.time()  # End of event stage

            # Accumulate stage durations
            sum_wait += t1 - t0
            sum_det += t2 - t1
            sum_track += t3 - t2
            sum_draw += t4 - t3
            sum_encode += t5 - t4
            sum_event += t6 - t5

            # Report average latency every 1000 frames
            if frame_count == 1000:
                print(
                    f"[{time.strftime('%m/%d/%Y-%H:%M:%S', time.localtime())}] VideoPipe: {self.input_url}, "
                    f"Det: {sum_det:.2f}ms, "
                    f"Track: {sum_track:.2f}ms, "
                    f"Draw: {sum_draw:.2f}ms, "
                    f"Encode: {sum_encode:.2f}ms, "
                    f"Event: {sum_event:.2f}ms, "
                    f"Wait: {sum_wait:.2f}ms "
                )
                # Reset counters
                frame_count = 0
                sum_det = 0
                sum_track = 0
                sum_draw = 0
                sum_encode = 0
                sum_event = 0
                sum_wait = 0


if __name__ == "__main__":
    # You can move this list into a separate YAML file and load it with PyYAML or similar.
    # Example:
    #   import yaml
    #   with open("streams.yaml", "r", encoding="utf-8") as f:
    #       args = yaml.safe_load(f)
    args = [
        {
            "gpu": 0,
            "input_url": "rtsp://172.16.3.210:8554/live/172.16.3.107",
            "output_url": "rtmp://172.16.3.210:1935/live/test_outq1",
        },
        {
            "gpu": 0,
            "input_url": "rtsp://172.16.3.210:8554/live/172.16.3.107",
            "output_url": "rtmp://172.16.3.210:1935/live/test_outq2",
        },
        {
            "gpu": 0,
            "input_url": "rtsp://172.16.3.210:8554/live/172.16.3.107",
            "output_url": "rtmp://172.16.3.210:1935/live/test_outq3",
        },
        {
            "gpu": 0,
            "input_url": "rtsp://172.16.3.210:8554/live/172.16.3.107",
            "output_url": "rtmp://172.16.3.210:1935/live/test_outq4",
        },
    ]

    # Use the 'spawn' start method to avoid CUDA context inheritance issues,
    # ensuring that each subprocess initializes CUDA independently.
    # When used with NVIDIA MPS (Multi-Process Service), spawn mode enables
    # multiple processes to share the same GPU compute resources, improving concurrency efficiency.
    torch.multiprocessing.set_start_method("spawn")
    process_pool = []
    for i in args:
        vp = VideoPipe(i["gpu"], i["input_url"], i["output_url"])
        vp.start()
        process_pool.append(vp)

    for vp in process_pool:
        vp.join()
