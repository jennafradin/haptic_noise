"""
WORK IN PROGRESS
Script for running the visuohaptic experiment.
"""

import csv
import logging
import os.path
import time
from typing import Any, Optional, Union

import cv2
import numpy as np
import requests
import torch
import wx
from icrpy_utils import vision
from tqdm import tqdm

logger = logging.getLogger("visiohaptic_incongruency")
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

"""
↓ Experiment parameters
"""

VISUAL_CONDITIONS = {
    "A": "./visual_conditions/A.png",
    "B": "./visual_conditions/B.png",
    "C": "./visual_conditions/C.png",
    "D": "./visual_conditions/D.png",
}
BETWEEN_TRIALS_S = 3
TRIAL_DURATION_S = 15

"""
↑ Experiment parameters
"""

MODEL_PATH = "torchscript_mobilenetv2_fp32.pth"
MODEL_URL = "https://github.com/PeterL1n/BackgroundMattingV2/releases/download/v1.0.0/torchscript_mobilenetv2_fp32.pth"

DEVICE = torch.device("cpu")
# DEVICE = torch.device("cuda")
# DTYPE = torch.float16
DTYPE = torch.float32

CWD = os.getcwd()


def get_model():
    if not os.path.exists(MODEL_PATH):
        logger.error("Could not find model at %s", MODEL_PATH)
        logger.info("Downloading model from GitHub")

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


def im_to_torch(src: np.ndarray, device, dtype) -> torch.Tensor:
    return (
        torch.tensor(src[np.newaxis, :], device=device, dtype=dtype).permute(0, 3, 1, 2)
        / 255
    )


def mask_image(mask, src, bgr):
    # mask = np.stack([to_show, to_show, to_show], axis=-1)
    # fg = np.where(to_show > 0.2, src, np.zeros_like(src))  # black background
    # fg = np.where(mask > 0.2, src, bgr_replace)  # image background
    # return fg

    mask_ind = mask > 0.2
    fg = np.copy(bgr)
    fg[mask_ind] = src[mask_ind]
    return fg


def do_trial(
    cap: cv2.VideoCapture, model: Any, bg: str, trial_duration_s: float
) -> Optional[str]:
    bg_im = cv2.imread(bg)
    bg_torch = im_to_torch(bg_im, DEVICE, DTYPE)

    t0 = time.time()

    while time.time() - t0 < trial_duration_s:
        ret, frame = cap.read()
        if not ret:
            logger.error("invalid frame received")
            continue

        frame_torch = im_to_torch(frame, DEVICE, DTYPE)
        pha = model(frame_torch, bg_torch)[0]

        pha_im: np.ndarray = pha[0][0].cpu().numpy()
        cv2.imshow("pha", pha_im)

        fg = mask_image(pha_im, frame, bg_im)
        cv2.imshow("fg", fg)

        if cv2.waitKey(1) == ord("q"):
            return None

    return None


def trial_transition(im_shape: Union[tuple, np.ndarray], transition_time: float):
    #! also change textures

    t0 = time.time()

    black_im = np.ones(im_shape)

    while time.time() - t0 < transition_time:
        cv2.imshow("transition", black_im)

        if cv2.waitKey(1) == ord("q"):
            return None


def main() -> int:
    # for cond, path in VISUAL_CONDITIONS.items():
    #     if not os.path.exists(path):
    #         logger.error("Could not find file %s for condition %s", path, cond)
    #         return 1

    # ask the user what file to use for the experiment
    filename: str

    _ = wx.App()
    frm = wx.Frame(None, title="Haptic Noise")
    with wx.FileDialog(
        frm,
        "Open CSV file",
        wildcard="CSV files (*.csv)|*.csv|All files (*.*)|*.*",
        style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
    ) as fileDialog:
        if fileDialog.ShowModal() == wx.ID_CANCEL:
            return 1  # exit if no file was selected

        filename = fileDialog.GetPath()

    filename_ok = isinstance(filename, str) and os.path.exists(filename)
    if not filename_ok:
        logger.error("Incorrect file selected.")
        return 1

    logger.info("Selected file: %s", filename)

    with open(filename, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        exp_data = list(reader)

    logger.info("Loaded experiment data")

    model = get_model()
    logger.info("Loaded model")
    cap, im_shape = vision.cap()
    logger.info("Setup camera")

    # exp_data prev next and replay current
    curr_trial = 0
    n_trials = len(exp_data)

    exp_done = False
    while not exp_done:
        visual, haptic = exp_data[curr_trial]["visual"], exp_data[curr_trial]["haptic"]
        logger.info(
            "Trial %d/%d. Conditions: v: %s | h: %s",
            curr_trial + 1,
            n_trials,
            visual,
            haptic,
        )

        bg = VISUAL_CONDITIONS[visual]

        ret = do_trial(cap, model, bg, TRIAL_DURATION_S)
        if ret is None:
            logger.info("exited prematurely")
            return 1

        # get answer
        # write answer
        curr_trial += 1
        if curr_trial >= n_trials:
            exp_done = True
        else:
            ret = trial_transition(im_shape, BETWEEN_TRIALS_S)
            if ret is None:
                logger.info("exited prematurely")
                return 1

    logger.info("Experiment done")

    return 0


main()
