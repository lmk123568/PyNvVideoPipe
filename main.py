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
            self.input_url,  # 输入 URL 或文件路径
            enable_frame_skip=False,  # 是否跳帧
            output_width=1024,  # 解码输出宽度
            output_height=576,  # 解码输出高度
            enable_auto_reconnect=True,  # 是否启用自动重连
            reconnect_delay_ms=2000,  # 重连间隔（毫秒）
            max_reconnects=5,  # 最大重连次数，超过后放弃
            open_timeout_ms=5000,  # 打开流超时（毫秒）
            read_timeout_ms=5000,  # 读取数据包超时（毫秒）
            buffer_size=4 * 1024 * 1024,  # 4MB 缓冲区，用于抖动容忍
            max_delay_ms=200,  # 允许的最大解码延迟（毫秒）
            reorder_queue_size=4,  # B 帧重排队列长度
            decoder_threads=1,  # 解码线程数
            surfaces=3,  # 用于缓冲的 CUDA surface 数量
        )

        encoder = pvp.Encoder(
            output_url=self.output_url,  # 输出 URL 或文件路径
            width=decoder.get_width(),  # 编码输出宽度
            height=decoder.get_height(),  # 编码输出高度
            fps=25,  # 编码输出帧率
            codec="libx264",  # 编码使用的视频编码器
            bitrate=1000000,  # 编码目标码率（kbps）
        )

        det = pvp.Yolo26DetTRT(
            engine_path="./yolo26n_1x3x576x1024_fp16.engine",
            conf_thres=0.25,
            device_id=0,
        )

        # 下面示例展示如何串联更多 GPU 模型做进一步推理，
        # 且全程不把帧数据拷回 CPU。
        # 请确保各模型输入/输出都保持为 GPU 张量，以避免 CPU <-> GPU 传输。
        # 例如：
        # import ultralytics
        # cls = ultralytics.YOLO("./yolo26n-cls.engine").to("cuda")   # 在 GPU 上加载模型
        # seg = ultralytics.YOLO("./yolo26n-seg.engine").to("cuda")
        # pose = ultralytics.YOLO("./yolo26n-pose.engine").to("cuda")

        # 用于可视化的 Supervision 标注器和跟踪器
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
                # 读取下一帧解码结果；pts 为显示时间戳
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

            # 业务逻辑

            t6 = time.time()

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
    # 你可以将该列表移到独立的 YAML 文件中，并通过 PyYAML 等方式加载。
    # 示例：
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

    # 使用 'spawn' 启动方式可避免 CUDA 上下文继承问题，
    # 确保每个子进程独立初始化 CUDA。
    # 配合 NVIDIA MPS（多进程服务）时，spawn 模式可让
    # 多个进程共享同一块 GPU 的计算资源，提升并发效率。
    mp.set_start_method("spawn")
    process_pool = []
    for i in args:
        vp = Pipeline(i["gpu"], i["input_url"], i["output_url"])
        vp.start()
        process_pool.append(vp)

    for vp in process_pool:
        vp.join()
