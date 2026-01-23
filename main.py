import time
import os
import sys
import glob
import multiprocessing as mp

from yolo26.models import YOLO26DetTRT

sys.path.append(os.getcwd())

so_files = glob.glob(os.path.join(os.getcwd(), "build/lib.*/*.so"))
if so_files:
    sys.path.insert(0, os.path.dirname(os.path.abspath(so_files[0])))

import nv


class VideoPipe(mp.Process):
    def __init__(self, input_url, output_url):
        super().__init__()
        self.input_url = input_url
        self.output_url = output_url

    def run(self):

        decoder = nv.Decoder(
            self.input_url,
            enable_frame_skip=True,
            output_width=1024,
            output_height=576,
            enable_auto_reconnect=True,
            reconnect_delay_ms=10000,
            max_reconnects=0,
            open_timeout_ms=5000,
            read_timeout_ms=5000,
            buffer_size=4 * 1024 * 1024,
            max_delay_ms=200,
            reorder_queue_size=32,
        )

        encoder = nv.Encoder(
            output_url=self.output_url,
            width=decoder.get_width(),
            height=decoder.get_height(),
            fps=25.0,
            codec="h264",
            bitrate=2_000_000,
        )

        yolo = YOLO26DetTRT(
            weights="./yolo26/yolo26n_1x3x576x1024_fp16.engine",
            device="cuda:0",
            conf_thres=0.25,
        )

        frame_count = 0

        sum_wait = 0
        sum_det = 0
        sum_draw = 0
        sum_encode = 0

        while 1:
            t0 = time.time()

            try:
                frame, pts = decoder.next_frame()
            except:
                sys.exit(1)

            if frame is None or frame.numel() == 0:
                break

            frame_count += 1

            t1 = time.time()

            det_results = yolo(frame)

            t2 = time.time()

            # encoder.encode(frame)

            t3 = time.time()

            encoder.encode(frame, pts)

            t4 = time.time()

            sum_wait += t1 - t0
            sum_det += t2 - t1
            sum_draw += t3 - t2
            sum_encode += t4 - t3

            if frame_count == 1000:
                print(
                    f"[{time.strftime('%m/%d/%Y-%H:%M:%S', time.localtime())}] VideoPipe: {self.input_url}, "
                    f"Det: {sum_det:.2f}ms, "
                    f"Draw: {sum_draw:.2f}ms, "
                    f"Encode: {sum_encode:.2f}ms, "
                    f"Wait: {sum_wait:.2f}ms "
                )

                frame_count = 0
                sum_det = 0
                sum_draw = 0
                sum_encode = 0
                sum_wait = 0


if __name__ == "__main__":
    args = [
        {
            "input_url": "rtsp://172.16.3.210:8554/live/172.16.3.107",
            "output_url": "rtmp://172.16.3.210:1935/live/test_outq1",
        },
        {
            "input_url": "rtsp://172.16.3.210:8554/live/172.16.3.107",
            "output_url": "rtmp://172.16.3.210:1935/live/test_outq2",
        },
        {
            "input_url": "rtsp://172.16.3.210:8554/live/172.16.3.107",
            "output_url": "rtmp://172.16.3.210:1935/live/test_outq3",
        },
        {
            "input_url": "rtsp://172.16.3.210:8554/live/172.16.3.107",
            "output_url": "rtmp://172.16.3.210:1935/live/test_outq4",
        },
    ]

    mp.set_start_method("spawn")
    process_pool = []
    for i in args:
        vp = VideoPipe(i["input_url"], i["output_url"])
        vp.start()
        process_pool.append(vp)

    for vp in process_pool:
        vp.join()
