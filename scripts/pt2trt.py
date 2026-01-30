import argparse
import json
import os
import sys

import torch


def parser_args() -> argparse.Namespace:
    print(
        """
==========================================================================
🚀 Ultralytics YOLO to TensorRT Conversion Script
   
   --weights  : Path to Ultralytics YOLO model (.pt)
   --fp16     : Enable FP16 precision. Default: False
   --device   : CUDA Device. Default: cuda:0
===========================================================================
          """
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")

    return parser.parse_args()


if __name__ == "__main__":
    args = parser_args()

    print("==> Check Environment")

    import tensorrt

    print(
        f"[\033[92mINFO\033[0m] Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    print(f"       Torch: {torch.__version__}")
    print(f"       CUDA: {torch.version.cuda}")
    print(
        f"       GPU Devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
    )
    print(f"       Tensorrt: {tensorrt.__version__}")

    print("\n==> Get Class Names")

    from ultralytics import YOLO

    pt_model = YOLO(args.weights)

    print(f"[\033[92mINFO\033[0m] {pt_model.names}")
    json_str = json.dumps(pt_model.names, ensure_ascii=False)
    with open("class_names.json", "w", encoding="utf-8") as file:
        file.write(json_str)

    f0 = args.weights[:-3]

    if args.fp16:
        f = f"{f0}_1x3x576x1024_fp16.engine"
    else:
        f = f"{f0}_1x3x576x1024_fp32.engine"

    if not os.path.exists(f):
        print("\n==> Exporting TRT Engine")

        print("[\033[92mINFO\033[0m] Args Parameters:")
        print(f"       - weights: {args.weights}")
        print(f"       - fp16:    {args.fp16}")
        print(f"       - device:  {args.device}")

        pt_model.export(
            format="engine",
            batch=1,
            imgsz=(576, 1024),
            half=args.fp16,
            workspace=None,
            device=torch.device(args.device),
        )

        os.rename(f0 + ".engine", f)

    import cv2
    import numpy as np

    img_bgr = cv2.imread("assets/zidane.jpg")
    img_bgr = cv2.resize(img_bgr, (1024, 576))

    # ultralytics
    trt_model = YOLO(f, task="detect")
    results = trt_model.predict(
        img_bgr, device=torch.device(args.device), half=args.fp16, imgsz=(576, 1024)
    )
    for r in results:
        trt_bboxes = r.boxes.xyxy.cpu().numpy()

    # our
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import nv_accel

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = torch.as_tensor(img_rgb.copy(), device=torch.device(args.device))
    our_model = nv_accel.Yolo26DetTRT(engine_path=f, conf_thres=0.25, device_id=0)
    our_bboxes = our_model(img_rgb)[:, :4].cpu().numpy()

    print(f"==> trt_bboxes:\n {trt_bboxes}")
    print(f"==> our_bboxes:\n {our_bboxes}")
