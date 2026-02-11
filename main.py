import os
import time

import supervision as sv
import torch
import torch.multiprocessing as mp


class Pipeline(mp.Process):
    def __init__(self, gpu, input_url, output_url):
        super().__init__()
        self.gpu = gpu
        self.input_url = input_url
        self.output_url = output_url

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    def run(self):
        # Import hardware-accelerated codec and YOLO detection model
        import pvp

        # Initialize CUDA-based video decoder with desired resolution and reconnection settings
        decoder = pvp.Decoder(
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
        )

        # Initialize CUDA-based H.264 encoder matching decoder's resolution
        fps = decoder.get_fps()
        fps = float(round(fps)) if fps and fps > 0 else 25.0
        encoder = pvp.Encoder(
            output_url=self.output_url,
            width=decoder.get_width(),
            height=decoder.get_height(),
            fps=fps,
            codec="h264",
            bitrate=1_000_000,  # 1 Mbps target bitrate
        )

        # Load TensorRT-optimized YOLOv26 nano model for object detection
        det = pvp.Yolo26DetTRT(
            engine_path="./yolo26n_1x3x576x1024_fp16.engine",
            conf_thres=0.25,  # Confidence threshold for detections
        )

        # The following example shows how to chain additional GPU models for further inference
        # without ever copying frame data back to the CPU.
        # Make sure every model’s inputs/outputs stay as GPU tensors to avoid any CPU <-> GPU transfers.
        # For instance:
        # import ultralytics
        # cls = ultralytics.YOLO("./yolo26n-cls.engine").to("cuda")   # load model on GPU
        # seg = ultralytics.YOLO("./yolo26n-seg.engine").to("cuda")
        # pose = ultralytics.YOLO("./yolo26n-pose.engine").to("cuda")

        # Supervision annotators and tracker for visualization
        tracker = sv.ByteTrack()
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        trace_annotator = sv.TraceAnnotator()

        # Frame counter for profiling
        frame_count = 0

        # timing accumulators for profiling each pipeline stage
        sum_wait = 0
        sum_det = 0
        sum_track = 0
        sum_draw = 0
        sum_encode = 0
        sum_event = 0

        try:
            while 1:
                t0 = time.time()  # Start timing for frame-wait stage

                # Fetch next decoded frame; pts is presentation timestamp
                try:
                    frame, pts = decoder.next_frame()
                except Exception as e:
                    print(f"Error: {e}")
                    continue

                if pts < 0 or frame.numel() == 0:
                    break

                frame_count += 1
                t1 = time.time()  # End of wait stage, start of detection

                # Run YOLO inference on GPU tensor
                det_results = det(frame)

                # Pass the GPU-resident frame directly at inference:
                # cls_results = cls(frame, verbose=False)[0].boxes.data        # remains on GPU
                # seg_results = seg(frame, verbose=False)[0].masks.data
                # pose_results = pose(frame, verbose=False)[0].keypoints.data

                t2 = time.time()  # End of detection stage

                # # Convert GPU tensor to numpy and build supervision.Detections
                det_results = det_results.cpu().numpy()
                det_results = sv.Detections(
                    xyxy=det_results[:, :4],
                    confidence=det_results[:, 4],
                    class_id=det_results[:, 5].astype(int),
                )
                # Update tracker with current detections
                tracker_results = tracker.update_with_detections(det_results)
                t3 = time.time()  # End of tracking stage

                # Move frame back to CPU for annotation
                annotated_frame = frame.cpu().numpy()

                # Build label text: tracker_id + class_id
                labels = [
                    f"#{tracker_id} {class_id}"
                    for tracker_id, class_id in zip(
                        tracker_results.tracker_id, tracker_results.class_id
                    )
                ]

                # Draw boxes, traces and labels on the frame
                annotated_frame = box_annotator.annotate(
                    scene=annotated_frame, detections=tracker_results
                )
                annotated_frame = trace_annotator.annotate(
                    scene=annotated_frame, detections=tracker_results
                )
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=tracker_results, labels=labels
                )
                t4 = time.time()  # End of drawing stage

                # Send annotated frame back to GPU and encode
                annotated_frame = torch.from_numpy(annotated_frame).to("cuda")
                encoder.encode(annotated_frame)
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
                        f"[{time.strftime('%m/%d/%Y-%H:%M:%S', time.localtime())}] Pipeline: {self.input_url}, "
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
        finally:
            encoder.finish()


if __name__ == "__main__":
    # You can move this list into a separate YAML file and load it with PyYAML or similar.
    # Example:
    #   import yaml
    #   with open("streams.yaml", "r", encoding="utf-8") as f:
    #       args = yaml.safe_load(f)
    args = [
        {
            "gpu": 0,
            "input_url": "rtsp://127.0.0.1:8554/live/input",
            "output_url": "rtmp://127.0.0.1:1935/live/out",
        },
        {
            "gpu": 0,
            "input_url": "rtsp://127.0.0.1:8554/live/input",
            "output_url": "output_annotated.mp4",
        },
        {
            "gpu": 0,
            "input_url": "input.mp4",
            "output_url": "output_annotated.mp4",
        },
        {
            "gpu": 0,
            "input_url": "input.mp4",
            "output_url": "rtmp://127.0.0.1:1935/live/out",
        },
    ]

    # Use the 'spawn' start method to avoid CUDA context inheritance issues,
    # ensuring that each subprocess initializes CUDA independently.
    # When used with NVIDIA MPS (Multi-Process Service), spawn mode enables
    # multiple processes to share the same GPU compute resources, improving concurrency efficiency.
    mp.set_start_method("spawn")
    process_pool = []
    for i in args:
        vp = Pipeline(i["gpu"], i["input_url"], i["output_url"])
        vp.start()
        process_pool.append(vp)

    for vp in process_pool:
        vp.join()
