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

        import pvp

        decoder = pvp.Decoder(
            self.input_url,  # Input URL or file path
            enable_frame_skip=False,  # Whether to skip frames
            output_width=1024,  # Output width for video decoder
            output_height=576,  # Output height for video decoder
            enable_auto_reconnect=True,  # Whether to enable auto reconnection
            reconnect_delay_ms=2000,  # Delay between reconnections in milliseconds
            max_reconnects=3,  # Maximum number of reconnections before giving up
            open_timeout_ms=5000,  # Timeout for opening the stream
            read_timeout_ms=5000,  # Timeout for reading packets
            buffer_size=4 * 1024 * 1024,  # 4MB buffer for jitter tolerance
            max_delay_ms=200,  # Max allowed decoding delay
            reorder_queue_size=4,  # B-frame reorder queue length
            decoder_threads=1,  # Number of decoder threads
            surfaces=3,  # Number of CUDA surfaces for buffering
        )

        encoder = pvp.Encoder(
            output_url=self.output_url,  # Output URL or file path
            width=decoder.get_width(),  # Output width for video encoder
            height=decoder.get_height(),  # Output height for video encoder
            fps=25,  # Output frame rate for video encoder
            codec="libx264",  # Video codec for encoding
            bitrate=1000000,  # Target bitrate for encoding in kbps
        )

        det = pvp.Yolo26DetTRT(
            engine_path="./yolo26n_1x3x576x1024_fp16.engine",
            conf_thres=0.25,
            device_id=0,
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

        frame_count = 0
        sum_wait = 0
        sum_det = 0
        sum_track = 0
        sum_draw = 0
        sum_encode = 0
        sum_event = 0

        while 1:
            t0 = time.time()

            
            try:
                # Fetch next decoded frame; pts is presentation timestamp
                frame, pts = decoder.next_frame()
            except Exception as e:
                if str(self.input_url).startswith(("rtsp://")):
                    print(f"[main.py] {self.input_url} 解码异常，54秒后重新拉流解码")
                    time.sleep(54)
                    continue
                print(f"[main.py] {self.input_url} 解码异常，进程退出")
                break

            frame_count += 1

            t1 = time.time()

            try:
                det_results = det(frame)

                t2 = time.time()

                det_results = det_results.cpu().numpy()
                det_results = sv.Detections(
                    xyxy=det_results[:, :4],
                    confidence=det_results[:, 4],
                    class_id=det_results[:, 5].astype(int),
                )
                tracker_results = tracker.update_with_detections(det_results)
                t3 = time.time()

                annotated_frame = frame.cpu().numpy()

                labels = [
                    f"#{tracker_id} {class_id}"
                    for tracker_id, class_id in zip(
                        tracker_results.tracker_id, tracker_results.class_id
                    )
                ]

                annotated_frame = box_annotator.annotate(
                    scene=annotated_frame, detections=tracker_results
                )
                annotated_frame = trace_annotator.annotate(
                    scene=annotated_frame, detections=tracker_results
                )
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=tracker_results, labels=labels
                )
                t4 = time.time()

                annotated_frame = torch.from_numpy(annotated_frame)
                encoder.encode(annotated_frame, pts)
                t5 = time.time()

                t6 = time.time()

            except Exception as e:
                print(f"[main.py] {self.input_url} 当前帧推理异常, 已跳过: {e}")
                continue

            sum_wait += t1 - t0
            sum_det += t2 - t1
            sum_track += t3 - t2
            sum_draw += t4 - t3
            sum_encode += t5 - t4
            sum_event += t6 - t5

            if frame_count == 1000:
                print(
                    f"[{time.strftime('%m/%d/%Y-%H:%M:%S', time.localtime())}] {self.input_url}, "
                    f"Det: {sum_det:.2f}ms, "
                    f"Track: {sum_track:.2f}ms, "
                    f"Draw: {sum_draw:.2f}ms, "
                    f"Encode: {sum_encode:.2f}ms, "
                    f"Event: {sum_event:.2f}ms, "
                    f"Wait: {sum_wait:.2f}ms "
                )
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
