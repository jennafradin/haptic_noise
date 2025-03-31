"""
Test script for removing background from images. Takes a picture at the start
and uses it as background for the rest of execution.
"""

import os
import time

import cv2
import numpy as np
import requests
import torch
import tqdm
from icrpy_utils import vision

# algorithm source
# https://github.com/PeterL1n/BackgroundMattingV2

NS_TO_MS = 1 / 1_000_000
MS_TO_S = 1 / 1_000

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print("using device:", DEVICE)
# DTYPE = torch.float16
DTYPE = torch.float32  # must be in accordance with the selected model: fp16 or fp32

MODEL_PATH = "torchscript_mobilenetv2_fp32.pth"
MODEL_URL = "https://github.com/PeterL1n/BackgroundMattingV2/releases/download/v1.0.0/torchscript_mobilenetv2_fp32.pth"

CWD = os.getcwd()


def get_model():
    if not os.path.exists(MODEL_PATH):
        print("Could not find model at %s", MODEL_PATH)
        print("Downloading model from GitHub")

        response = requests.get(MODEL_URL, stream=True)

        with open(MODEL_PATH, "wb") as handle:
            for data in tqdm(response.iter_content()):
                handle.write(data)

    # model = torch.jit.load('torchscript_mobilenetv2_fp16.pth')
    model = torch.jit.load(MODEL_PATH)
    model.backbone_scale = 0.25
    model.refine_mode = "sampling"
    model.refine_sample_pixels = 80_000

    model = model.to(DEVICE)

    return model


def transform(src: np.ndarray, device, dtype) -> torch.Tensor:
    """Take an image from OpenCV and turn it into a torch tensor

    Args:
        src (np.ndarray): input OpenCV image
        device: PyTorch device to save the tensor to, as defined above
        dtype: PyTorch data type as defined above

    Returns:
        torch.Tensor: _description_
    """
    return (
        torch.tensor(src[np.newaxis, :], device=device, dtype=dtype).permute(0, 3, 1, 2)
        / 255
    )


def mask_image(mask, src, bgr):
    """Replace pixels in the background with pixels in the foreground,
    according to the mask.

    Args:
        mask: Array with values between 0 and 1. Pixels at 0 will be
        treated as background, pixels above 0 will be treated as foreground
        src: foreground image
        bgr: background image

    Returns:
        _type_: _description_
    """

    mask_ind = mask > 0.2
    fg = np.copy(bgr)
    fg[mask_ind] = src[mask_ind]
    return fg


def main() -> int:
    model = get_model()

    # create video capture object
    cap, im_shape = vision.cap()
    # grab a frame which will act as background
    _, bgr = cap.read()

    bgr_torch = transform(bgr, DEVICE, DTYPE)  # copy to use with model
    bgr_replace = np.copy(bgr)  # copy to use when masking

    t_start = time.perf_counter_ns()
    i = 0
    print("start")
    while cap.isOpened():
        i += 1

        # grab current frame
        _, src = cap.read()
        src_torch = transform(src, DEVICE, DTYPE)

        # inference: detect foreground
        pha = model(src_torch, bgr_torch)[0]

        # print fps every 100 frames
        if i % 100 == 0:
            t_100 = time.perf_counter_ns()
            d_t = t_100 - t_start
            fps = 100 / (d_t / 1000000000)
            print(f"{fps=}")
            t_start = time.perf_counter_ns()

        # extract mask array from inference result
        to_show = pha[0][0].cpu().numpy()
        cv2.imshow("pha", to_show)  # show raw mask

        fg = mask_image(to_show, src, bgr_replace)
        cv2.imshow("fg", fg)  # show masked image

        if cv2.waitKey(1) == ord("q"):
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
